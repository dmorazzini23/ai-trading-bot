from __future__ import annotations
from datetime import UTC, date, datetime, timedelta
import os
from typing import Any
from zoneinfo import ZoneInfo
from ai_trading.utils.lazy_imports import load_pandas
from ai_trading.config import get_settings
from ai_trading.data.market_calendar import previous_trading_session, rth_session_utc
from ai_trading.data.fetch import get_bars, get_minute_df
from ai_trading.data.fetch import get_bars as http_get_bars
from ai_trading.logging import get_logger
from ai_trading.logging.empty_policy import classify as _empty_classify
from ai_trading.logging.empty_policy import record as _empty_record
from ai_trading.logging.empty_policy import should_emit as _empty_should_emit
from ai_trading.logging.emit_once import emit_once
from ai_trading.logging.normalize import canon_feed as _canon_feed
from ai_trading.logging.normalize import canon_symbol as _canon_symbol
from ai_trading.logging.normalize import canon_timeframe as _canon_tf
from ai_trading.utils.time import now_utc
from .timeutils import ensure_utc_datetime, expected_regular_minutes
from .models import StockBarsRequest, TimeFrame
from ._alpaca_guard import should_import_alpaca_sdk
from .fetch.sip_disallowed import sip_disallowed
import time


class _FallbackAPIError(Exception):
    """Fallback APIError when Alpaca SDK is unavailable."""


APIError = _FallbackAPIError

if should_import_alpaca_sdk():  # pragma: no cover - alpaca optional
    try:
        from alpaca.common.exceptions import APIError as _RealAPIError
    except (ImportError, AttributeError):
        pass
    else:
        APIError = _RealAPIError

__all__ = ["TimeFrame", "StockBarsRequest"]

# Lazy pandas proxy; only imported on first use
pd = load_pandas()

_log = get_logger(__name__)
'AI-AGENT-REF: canonicalizers moved to ai_trading.logging.normalize'

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in _TRUTHY


def _default_sip_entitled() -> bool:
    explicit = _env_bool("ALPACA_SIP_ENTITLED")
    if explicit is not None:
        return explicit
    legacy = _env_bool("ALPACA_HAS_SIP")
    if legacy is not None:
        return legacy
    return False


def get_alpaca_feed(prefer_sip: bool, *, sip_entitled: bool | None = None) -> str:
    """Return the Alpaca feed name to use given SIP entitlement state."""

    entitled = sip_entitled
    if entitled is None:
        entitled = _default_sip_entitled()
    return "sip" if prefer_sip and entitled else "iex"

def _format_fallback_payload(tf_str: str, feed_str: str, start_utc: datetime, end_utc: datetime) -> list[str]:
    s = start_utc.astimezone(UTC).isoformat()
    e = end_utc.astimezone(UTC).isoformat()
    return [tf_str, feed_str, s, e]

def _log_fallback_window_debug(logger, day_et: date, start_utc: datetime, end_utc: datetime) -> None:
    try:
        logger.debug(
            'DATA_FALLBACK_WINDOW_DEBUG',
            extra={
                'et_day': day_et.isoformat(),
                'rth_et': '09:30-16:00',
                'rth_utc': f'{start_utc.astimezone(UTC).isoformat()}..{end_utc.astimezone(UTC).isoformat()}',
            },
        )
    except (ValueError, TypeError):
        pass

# Fallback shims removed: TimeFrame and StockBarsRequest now come from alpaca.data
COMMON_EXC = (ValueError, KeyError, AttributeError, TypeError, RuntimeError, ImportError, OSError, ConnectionError, TimeoutError)

def _ensure_df(obj: Any) -> pd.DataFrame:
    """Best-effort conversion to DataFrame, never raises."""
    try:
        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, 'df'):
            df = getattr(obj, 'df', None)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return pd.DataFrame(obj) if obj is not None else pd.DataFrame()
    except (ValueError, TypeError):
        return pd.DataFrame()

def empty_bars_dataframe() -> pd.DataFrame:
    cols = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
    return pd.DataFrame(columns=cols)

def _create_empty_bars_dataframe(timeframe: str | None = None) -> pd.DataFrame:
    """Return an empty OHLCV DataFrame including a timestamp column."""

    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame({col: [] for col in cols})
    return df

def _is_minute_timeframe(tf) -> bool:
    try:
        return str(tf).lower() in ('1min', '1m', 'minute', '1 minute')
    except (ValueError, TypeError):
        return False


_ENTITLE_CACHE: dict[int, tuple[float, set[str]]] = {}
_ENTITLE_TTL = 300

def _get_entitled_feeds(client: Any) -> set[str]:
    """Return set of feeds the account is entitled to."""
    now = time.time()
    key = id(client)
    cached = _ENTITLE_CACHE.get(key)
    if cached and (now - cached[0] < _ENTITLE_TTL):
        return cached[1]
    feeds: set[str] = {"iex"}
    get_acct = getattr(client, "get_account", None)
    if callable(get_acct):
        try:
            acct = get_acct()
            sub = getattr(acct, "market_data_subscription", None) or getattr(acct, "data_feed", None)
            if isinstance(sub, str):
                feeds = {sub.lower()}
            elif isinstance(sub, (set, list, tuple)):
                feeds = {str(x).lower() for x in sub}
        except COMMON_EXC as e:  # pragma: no cover - network
            _log.debug('FEED_ENTITLE_CHECK_FAIL', extra={'error': str(e)})
    _ENTITLE_CACHE[key] = (now, feeds)
    return feeds

def _ensure_entitled_feed(client: Any, requested: str) -> str:
    """Return a feed we are entitled to, falling back when necessary."""
    feeds = _get_entitled_feeds(client)
    req_raw = str(requested or '').lower()
    normalized_req = req_raw.replace("alpaca_", "")
    sip_allowed = not sip_disallowed()
    env_flag = _env_bool("ALPACA_SIP_ENTITLED")
    if env_flag is None:
        env_flag = _env_bool("ALPACA_HAS_SIP")
    sip_capability = "sip" in feeds
    sip_entitled_flag = False
    if sip_allowed and sip_capability:
        if env_flag is False:
            sip_entitled_flag = False
        else:
            sip_entitled_flag = True
    prefer_sip = normalized_req == "sip"
    resolved = get_alpaca_feed(prefer_sip, sip_entitled=sip_entitled_flag)
    if resolved in feeds:
        return resolved
    eligible_feeds = [feed for feed in feeds if feed != "sip" or sip_entitled_flag]
    if eligible_feeds:
        alt = eligible_feeds[0]
        if resolved != alt:
            _log.warning(
                'ALPACA_FEED_UNENTITLED_SWITCH',
                extra={'requested': normalized_req, 'using': alt},
            )
        return alt
    emit_once(
        _log,
        f'no_feed:{normalized_req}',
        'error',
        'ALPACA_FEED_UNENTITLED',
        requested=normalized_req,
    )
    return resolved

def _client_fetch_stock_bars(client: Any, request: "StockBarsRequest"):
    """Call the appropriate Alpaca SDK method to fetch bars."""
    get_stock_bars_fn = getattr(client, "get_stock_bars", None)
    if callable(get_stock_bars_fn):
        return get_stock_bars_fn(request)
    get_bars_fn = getattr(client, "get_bars", None)
    if not callable(get_bars_fn):
        raise AttributeError("Alpaca client missing get_stock_bars/get_bars")
    params = {}
    if getattr(request, "start", None) is not None:
        params["start"] = ensure_utc_datetime(request.start).isoformat()
    if getattr(request, "end", None) is not None:
        params["end"] = ensure_utc_datetime(request.end).isoformat()
    if getattr(request, "limit", None) is not None:
        params["limit"] = request.limit
    if getattr(request, "feed", None) is not None:
        params["feed"] = request.feed
    return get_bars_fn(request.symbol_or_symbols, request.timeframe, **params)

def safe_get_stock_bars(client: Any, request: "StockBarsRequest", symbol: str, context: str='') -> pd.DataFrame:
    """
    Safely fetch stock bars via Alpaca client and always return a DataFrame.
    This is a faithful move of the original implementation from bot_engine,
    with identical behavior and logging fields.
    """
    from .models import TimeFrame, StockBarsRequest
    symbol = _canon_symbol(symbol)
    sym_attr = getattr(request, 'symbol_or_symbols', None)
    try:
        if isinstance(sym_attr, list):
            request.symbol_or_symbols = [_canon_symbol(sym_attr[0])]
        elif isinstance(sym_attr, str):
            request.symbol_or_symbols = _canon_symbol(sym_attr)
    except (TypeError, AttributeError, ValueError):
        pass
    now = now_utc()
    prev_open, _ = rth_session_utc(previous_trading_session(now.date()))
    end_dt = ensure_utc_datetime(getattr(request, 'end', None) or now, default=now, clamp_to='eod', allow_callables=False)
    start_dt = ensure_utc_datetime(getattr(request, 'start', None) or prev_open, default=prev_open, clamp_to='bod', allow_callables=False)
    iso_start = start_dt.isoformat()
    iso_end = end_dt.isoformat()
    try:
        request.start = start_dt
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        request.end = end_dt
    except (AttributeError, TypeError, ValueError):
        pass
    feed_req = _canon_feed(getattr(request, 'feed', None))
    if feed_req:
        request.feed = _ensure_entitled_feed(client, feed_req)
    try:
        try:
            response = _client_fetch_stock_bars(client, request)
        except APIError as e:
            _log.error(
                "ALPACA_BARS_APIERROR",
                extra={"symbol": symbol, "context": context, "error": str(e)},
            )
            return _create_empty_bars_dataframe()
        except (ValueError, TypeError) as e:
            status = getattr(e, 'status_code', None)
            if status in (401, 403):
                feed_str = _canon_feed(getattr(request, 'feed', None))
                alt = _ensure_entitled_feed(client, feed_str)
                if alt != feed_str:
                    request.feed = alt
                    time.sleep(0.25)
                    response = _client_fetch_stock_bars(client, request)
                else:
                    emit_once(_log, f'{symbol}:{feed_str}', 'error', 'ALPACA_BARS_UNAUTHORIZED', symbol=symbol, context=context, feed=feed_str)
                    raise
            else:
                raise
        df = _ensure_df(getattr(response, 'df', response))
        if df.empty:
            now_ts = datetime.now(UTC)
            key = (
                symbol,
                str(context),
                _canon_feed(getattr(request, 'feed', None)),
                _canon_tf(getattr(request, 'timeframe', '')),
                now_ts.date().isoformat(),
            )
            if _empty_should_emit(key, now_ts):
                lvl = _empty_classify(is_market_open=False)
                cnt = _empty_record(key, now_ts)
                _log.log(lvl, 'ALPACA_BARS_EMPTY', extra={'symbol': symbol, 'context': context, 'occurrences': cnt})
            time.sleep(0.25)
            try:
                response = _client_fetch_stock_bars(client, request)
                df = _ensure_df(getattr(response, 'df', response))
            except APIError as e:
                _log.error(
                    "ALPACA_BARS_APIERROR",
                    extra={"symbol": symbol, "context": context, "error": str(e)},
                )
                df = pd.DataFrame()
            except COMMON_EXC:
                df = pd.DataFrame()
            if df.empty:
                tf_str = _canon_tf(getattr(request, 'timeframe', ''))
                feed_str = _canon_feed(getattr(request, 'feed', None))
                if tf_str.lower() in {'1day', 'day'}:
                    mdf = get_minute_df(symbol, iso_start, iso_end, feed=feed_str)
                    if mdf is not None and (not mdf.empty):
                        rdf = _resample_minutes_to_daily(mdf)
                        if rdf is not None and (not rdf.empty):
                            df = rdf
                        else:
                            df = pd.DataFrame()
                    else:
                        df = pd.DataFrame()
                    if df.empty:
                        try:
                            alt_req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, limit=2, feed=feed_str)
                            alt_resp = _client_fetch_stock_bars(client, alt_req)
                            df2 = _ensure_df(getattr(alt_resp, 'df', alt_resp))
                            if isinstance(df2.index, pd.MultiIndex):
                                df2 = df2.xs(symbol, level=0, drop_level=False).droplevel(0)
                            df2 = df2.sort_index()
                            if not df2.empty:
                                last = df2.index[-1]
                                if last.date() == start_dt.date():
                                    df = df2.loc[[last]]
                        except APIError as e:
                            _log.error(
                                "ALPACA_BARS_APIERROR",
                                extra={"symbol": symbol, "context": context, "error": str(e)},
                            )
                        except (ValueError, TypeError) as e:
                            status = getattr(e, 'status_code', None)
                            if status in (401, 403):
                                feed_str = _ensure_entitled_feed(client, feed_str)
                                emit_once(_log, f'{symbol}:{feed_str}', 'error', 'ALPACA_BARS_UNAUTHORIZED', symbol=symbol, context=context, feed=feed_str)
                                raise
                            _log.warning('ALPACA_LIMIT_FETCH_FAILED', extra={'symbol': symbol, 'context': context, 'error': str(e)})
                elif _is_minute_timeframe(tf_str):
                    df = get_minute_df(symbol, iso_start, iso_end, feed=feed_str)
                else:
                    df = http_get_bars(symbol, tf_str, iso_start, iso_end, feed=feed_str)
                if df is None or df.empty:
                    return _create_empty_bars_dataframe()
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0, drop_level=False).droplevel(0)
            except (KeyError, ValueError):
                return _create_empty_bars_dataframe()
        if not df.empty:
            return df
        _now = datetime.now(UTC)
        _key = (symbol, str(context), _canon_feed(getattr(request, 'feed', None)), _canon_tf(getattr(request, 'timeframe', '')), _now.date().isoformat())
        if _empty_should_emit(_key, _now):
            lvl = _empty_classify(is_market_open=False)
            cnt = _empty_record(_key, _now)
            _log.log(lvl, 'ALPACA_PARSE_EMPTY', extra={'symbol': symbol, 'context': context, 'feed': _canon_feed(getattr(request, 'feed', None)), 'timeframe': _canon_tf(getattr(request, 'timeframe', '')), 'occurrences': cnt})
        return pd.DataFrame()
    except COMMON_EXC as e:
        _log.error('ALPACA_BARS_FETCH_FAILED', extra={'symbol': symbol, 'context': context, 'error': str(e)})
        if _is_minute_timeframe(getattr(request, 'timeframe', '')):
            return _ensure_df(
                get_minute_df(symbol, iso_start, iso_end, feed=_canon_feed(getattr(request, 'feed', None)))
            )
        tf_str = _canon_tf(getattr(request, 'timeframe', ''))
        feed_str = _canon_feed(getattr(request, 'feed', None))
        df = http_get_bars(symbol, tf_str, iso_start, iso_end, feed=feed_str)
        return _ensure_df(df)

def _fetch_daily_bars(client, symbol, start, end, **kwargs):
    symbol = _canon_symbol(symbol)
    start = ensure_utc_datetime(start)
    end = ensure_utc_datetime(end)
    get_bars_fn = getattr(client, 'get_bars', None)
    if not callable(get_bars_fn):
        raise RuntimeError('Alpaca client missing get_bars()')
    try:
        return get_bars_fn(symbol, timeframe='1Day', start=start, end=end, **kwargs)
    except (ValueError, TypeError) as e:
        _log.exception('ALPACA_DAILY_FAILED', extra={'symbol': symbol, 'error': str(e)})
        raise

def _get_minute_bars(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    feed: str,
    adjustment: str | None = None,
) -> pd.DataFrame:
    symbol = _canon_symbol(symbol)
    try:
        df = get_bars(
            symbol=symbol,
            timeframe='1Min',
            start=start_dt,
            end=end_dt,
            feed=feed,
            adjustment=adjustment,
        )
    except (ValueError, TypeError):
        df = None
    if df is None or not hasattr(df, 'empty') or getattr(df, 'empty', True):
        return empty_bars_dataframe()
    return df

def _resample_minutes_to_daily(df, tz='America/New_York'):
    """Resample minute bars to daily OHLCV over regular trading hours."""
    if df is None or df.empty:
        return df
    try:
        mkt = df.copy()
        mkt = mkt.tz_convert(tz) if mkt.index.tz is not None else mkt.tz_localize(tz)
        mkt = mkt.between_time('09:30', '16:00', inclusive='both')
        o = mkt['open'].resample('1D').first()
        h = mkt['high'].resample('1D').max()
        l = mkt['low'].resample('1D').min()
        c = mkt['close'].resample('1D').last()
        v = mkt.get('volume')
        v = v.resample('1D').sum() if v is not None else None
        out = pd.concat({'open': o, 'high': h, 'low': l, 'close': c}, axis=1)
        if v is not None:
            out['volume'] = v
        out = out.dropna(how='all').tz_convert('UTC')
        return out
    except (ValueError, TypeError) as e:
        _log.warning('RESAMPLE_DAILY_FAILED', extra={'error': str(e)})
        return df

def get_daily_bars(symbol: str, client, start: datetime, end: datetime, feed: str | None=None):
    """Fetch daily bars; fallback to alternate feed then resampled minutes."""
    symbol = _canon_symbol(symbol)
    S = get_settings()
    if feed is None:
        feed = S.alpaca_data_feed
    adjustment = S.alpaca_adjustment
    start = ensure_utc_datetime(start)
    end = ensure_utc_datetime(end)
    df = _fetch_daily_bars(client, symbol, start, end, feed=feed, adjustment=adjustment)
    if df is not None and (not df.empty):
        return df
    alt = 'iex' if feed == 'sip' else 'sip'
    df = _fetch_daily_bars(client, symbol, start, end, feed=alt, adjustment=adjustment)
    if df is not None and (not df.empty):
        return df
    try:
        minutes_start = end - timedelta(days=5)
        mdf = _get_minute_bars(symbol, minutes_start, end, feed=feed, adjustment=adjustment)
        if mdf is not None and (not mdf.empty):
            rdf = _resample_minutes_to_daily(mdf)
            if rdf is not None and (not rdf.empty):
                _log.info('DAILY_FALLBACK_RESAMPLED', extra={'symbol': symbol, 'rows': len(rdf)})
                return rdf
    except (ValueError, TypeError) as e:
        _log.warning('DAILY_MINUTE_RESAMPLE_FAILED', extra={'symbol': symbol, 'error': str(e)})
    raise ValueError('empty_bars')

def _minute_fallback_window(now_utc: datetime) -> tuple[datetime, datetime]:
    """Compute NYSE session for the current or previous trading day."""
    today_ny = now_utc.astimezone(ZoneInfo('America/New_York')).date()
    start_u, end_u = rth_session_utc(today_ny)
    if now_utc < start_u or now_utc > end_u:
        prev_day = previous_trading_session(today_ny)
        start_u, end_u = rth_session_utc(prev_day)
    return (start_u, end_u)

def fetch_minute_fallback(client, symbol, now_utc: datetime) -> pd.DataFrame:
    symbol = _canon_symbol(symbol)
    now_utc = ensure_utc_datetime(now_utc)
    start_u, end_u = _minute_fallback_window(now_utc)
    day_et = start_u.astimezone(ZoneInfo('America/New_York')).date()
    _log_fallback_window_debug(_log, day_et, start_u, end_u)
    feed_str = 'iex'
    try:
        df = _get_minute_bars(symbol, start_u, end_u, feed=feed_str)
    except (KeyError, ValueError):
        df = empty_bars_dataframe()
    rows = len(df)
    if rows < 300:
        _log.warning('DATA_HEALTH_MINUTE_INCOMPLETE', extra={'rows': rows, 'expected': expected_regular_minutes(), 'start': start_u.astimezone(UTC).isoformat(), 'end': end_u.astimezone(UTC).isoformat(), 'feed': feed_str})
        try:
            df_sip = _get_minute_bars(symbol, start_u, end_u, feed='sip')
        except (KeyError, ValueError):
            df_sip = empty_bars_dataframe()
        if len(df_sip) > rows:
            df = df_sip
            feed_str = 'sip'
            rows = len(df)
    payload = _format_fallback_payload('1Min', feed_str, start_u, end_u)
    _log.info('DATA_FALLBACK_ATTEMPT', extra={'provider': 'alpaca', 'fallback': payload})
    if rows >= 300:
        _log.info('DATA_HEALTH: minute fallback ok', extra={'rows': rows})
    return df

def _parse_bars(payload: Any, symbol: str, tz: str) -> pd.DataFrame:
    if not payload:
        return empty_bars_dataframe()
    if isinstance(payload, dict):
        bars = payload.get('bars') or payload.get('data') or payload.get('results')
        if not bars:
            return empty_bars_dataframe()
        try:
            return _ensure_df(pd.DataFrame(bars))
        except (ValueError, TypeError):
            return empty_bars_dataframe()
    if isinstance(payload, pd.DataFrame):
        return payload
    return empty_bars_dataframe()
