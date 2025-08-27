from __future__ import annotations
import datetime as _dt
import os
import warnings
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo
import importlib
from ai_trading.utils.lazy_imports import load_pandas

try:
    from ai_trading.config import get_settings
except Exception:

    def get_settings():
        return None
from ai_trading.data.timeutils import ensure_utc_datetime
from ai_trading.logging.empty_policy import classify as _empty_classify
from ai_trading.logging.empty_policy import record as _empty_record
from ai_trading.logging.empty_policy import should_emit as _empty_should_emit
from ai_trading.logging.normalize import canon_feed as _canon_feed
from ai_trading.logging.normalize import canon_timeframe as _canon_tf
from ai_trading.logging.normalize import normalize_extra as _norm_extra
from ai_trading.logging import logger


# Optional dependency placeholders
pd: Any | None = None


class _RequestsModulePlaceholder:
    get = None


requests: Any = _RequestsModulePlaceholder()


class _YFinancePlaceholder:
    download = None


yf: Any = _YFinancePlaceholder()


class RequestException(Exception):
    """Fallback request exception when ``requests`` is missing."""


class Timeout(RequestException):
    pass


class ConnectionError(RequestException):
    pass


class HTTPError(RequestException):
    pass


def _incr(metric: str, *, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
    try:
        from ai_trading.monitoring import metrics as _metrics
        _metrics.incr(metric, value=value, tags=tags)
    except Exception:  # pragma: no cover - metrics optional
        pass


def _to_timeframe_str(tf: object) -> str:
    return _canon_tf(tf)


def _to_feed_str(feed: object) -> str:
    return _canon_feed(feed)


class DataFetchError(Exception):
    """Error raised when market data retrieval fails."""  # AI-AGENT-REF: stable public symbol


# Backwards compat alias
DataFetchException = DataFetchError


class FinnhubAPIException(Exception):
    """Minimal Finnhub API error for tests."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(str(status_code))


def ensure_datetime(value: Any) -> _dt.datetime:
    """Coerce various datetime inputs into timezone-aware UTC datetime.

    Rules for market-data windows:
    - If ``value`` is callable, call it (no args) and re-normalize.
    - If ``value`` is a *naive* ``datetime``, interpret it as **America/New_York**
      (exchange time) before converting to UTC.
    - Otherwise, delegate to ``ensure_utc_datetime``.
    """
    pd_mod = _ensure_pandas()
    out_of_bounds = ()
    if pd_mod is not None:
        try:
            out_of_bounds = (pd_mod.errors.OutOfBoundsDatetime,)
        except Exception:
            out_of_bounds = ()
    if callable(value):
        try:
            value = value()
        except (*out_of_bounds, TypeError, ValueError, AttributeError) as e:  # type: ignore[misc]
            raise TypeError(f'Invalid datetime input: {e}') from e
    if isinstance(value, _dt.datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo('America/New_York'))
    try:
        return ensure_utc_datetime(value, allow_callables=False)
    except (*out_of_bounds, TypeError, ValueError, AttributeError) as e:  # type: ignore[misc]
        raise TypeError(f'Invalid datetime input: {e}') from e


def _format_fallback_payload_df(tf_str: str, feed_str: str, start_dt: _dt.datetime, end_dt: _dt.datetime) -> list[str]:
    """UTC ISO payload for consistent logging."""
    s = ensure_datetime(start_dt).astimezone(UTC).isoformat()
    e = ensure_datetime(end_dt).astimezone(UTC).isoformat()
    return [tf_str, feed_str, s, e]
_MINUTE_CACHE: dict[str, tuple[int, int]] = {}

def get_cached_minute_timestamp(symbol: str) -> int | None:
    """Return cached last bar timestamp for symbol."""
    rec = _MINUTE_CACHE.get(symbol)
    return rec[0] if rec else None

def set_cached_minute_timestamp(symbol: str, ts_epoch_s: int) -> None:
    """Store last bar timestamp with current insertion time."""
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    _MINUTE_CACHE[symbol] = (int(ts_epoch_s), now_s)

def clear_cached_minute_timestamp(symbol: str) -> None:
    """Remove cached entry for symbol."""
    _MINUTE_CACHE.pop(symbol, None)

def age_cached_minute_timestamps(max_age_seconds: int) -> int:
    """Drop cache entries older than max_age_seconds (based on inserted time)."""
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    to_del = [sym for sym, (_, ins) in _MINUTE_CACHE.items() if now_s - ins > max_age_seconds]
    for sym in to_del:
        _MINUTE_CACHE.pop(sym, None)
    return len(to_del)

def last_minute_bar_age_seconds(symbol: str) -> int | None:
    """Age in seconds of last cached minute bar for symbol, or None if absent."""
    ts = get_cached_minute_timestamp(symbol)
    if ts is None:
        return None
    now_s = int(_dt.datetime.now(tz=UTC).timestamp())
    return max(0, now_s - int(ts))
_DEFAULT_FEED = 'iex'
_VALID_FEEDS = ('iex', 'sip')

class _FinnhubFetcherStub:
    """Minimal stub with a fetch() method; tests monkeypatch this."""
    is_stub = True

    def fetch(self, *args, **kwargs):
        raise NotImplementedError
fh_fetcher = _FinnhubFetcherStub()

def get_last_available_bar(symbol: str) -> pd.DataFrame:
    """Placeholder; tests monkeypatch this to return a last available daily bar."""
    raise NotImplementedError('Tests should monkeypatch get_last_available_bar')

def _default_window_for(timeframe: Any) -> tuple[_dt.datetime, _dt.datetime]:
    """Derive [start, end] when callers omit them."""
    now = _dt.datetime.now(tz=UTC)
    end = now - _dt.timedelta(minutes=1)
    tf = str(timeframe).lower()
    if 'day' in tf:
        days = int(os.getenv('DATA_LOOKBACK_DAYS_DAILY', '200'))
    else:
        days = int(os.getenv('DATA_LOOKBACK_DAYS_MINUTE', '5'))
    start = end - _dt.timedelta(days=days)
    return (start, end)

def _flatten_and_normalize_ohlcv(df: pd.DataFrame, symbol: str | None=None) -> pd.DataFrame:
    """Make YF/other OHLCV DataFrames uniform.

    - flatten MultiIndex columns
    - lower/snake columns
    - ensure 'close' exists (fallback to 'adj_close')
    - de-duplicate & sort index, convert index to UTC and tz-naive
    """
    pd = _ensure_pandas()
    if pd is None:
        return []  # type: ignore[return-value]
    if isinstance(df.columns, pd.MultiIndex):
        try:
            lvl0 = set(map(str, df.columns.get_level_values(0)))
            if {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'} & lvl0:
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = ['_'.join([str(x) for x in tup if x is not None]) for tup in df.columns]
        except (AttributeError, IndexError, TypeError):
            df.columns = ['_'.join([str(x) for x in tup if x is not None]) for tup in df.columns]
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    if 'close' not in df.columns and 'adj_close' in df.columns:
        df['close'] = df['adj_close']
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            tz = df.index.tz
            if tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
        except (AttributeError, TypeError, ValueError):
            pass
        df = df[~df.index.duplicated(keep='last')].sort_index()
    return df

def _yahoo_get_bars(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    """Return a DataFrame with a tz-aware 'timestamp' column between start and end."""
    pd = _ensure_pandas()
    yf = _ensure_yfinance()
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    if pd is None:
        return []  # type: ignore[return-value]
    if getattr(yf, "download", None) is None:
        idx = pd.DatetimeIndex([], tz='UTC', name='timestamp')
        cols = ['open', 'high', 'low', 'close', 'volume']
        return pd.DataFrame(columns=cols, index=idx).reset_index()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*auto_adjust.*', module='yfinance')
        df = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=False,
            threads=False,
            progress=False,
            group_by='column',
        )
    if df is None or df.empty:
        idx = pd.DatetimeIndex([], tz='UTC', name='timestamp')
        cols = ['open', 'high', 'low', 'close', 'volume']
        return pd.DataFrame(columns=cols, index=idx).reset_index()
    df = df.reset_index().rename(columns={df.index.name or 'Date': 'timestamp'})
    if 'timestamp' not in df.columns:
        for c in df.columns:
            if c.lower() in ('date', 'datetime'):
                df = df.rename(columns={c: 'timestamp'})
                break
    df = _flatten_and_normalize_ohlcv(df, symbol)
    return df

def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV DataFrame or return empty."""
    pd = _ensure_pandas()
    if pd is None:
        return []  # type: ignore[return-value]
    if df is None or getattr(df, 'empty', True):
        return pd.DataFrame()
    return _flatten_and_normalize_ohlcv(df)


def _ensure_http_client():
    try:
        from importlib import import_module

        client = import_module("ai_trading.utils.http")
        logger.debug('HTTP_INIT_PRIMARY', extra={'transport': 'ai_trading.utils.http'})
        return client
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug('HTTP_INIT_FALLBACK', extra={'transport': 'requests'})
        return None


def _ensure_pandas():
    global pd
    if pd is None:
        try:
            pd = load_pandas()
        except Exception:  # pragma: no cover - optional dependency
            pd = None
    return pd


def _ensure_yfinance():
    global yf
    if getattr(yf, "download", None) is None:
        try:
            import yfinance as _yf  # type: ignore
            try:
                if hasattr(_yf, 'set_tz_cache_location'):
                    os.makedirs('/tmp/py-yfinance', exist_ok=True)
                    _yf.set_tz_cache_location('/tmp/py-yfinance')
            except OSError:
                pass
            yf = _yf
        except ImportError:  # pragma: no cover - optional dependency
            yf = _YFinancePlaceholder()
            logger.info('YFINANCE_MISSING', extra={'hint': 'pip install yfinance'})
    return yf


def _ensure_requests():
    global requests, ConnectionError, HTTPError, RequestException, Timeout
    if getattr(requests, "get", None) is None:
        try:
            import requests as _requests  # type: ignore
            from requests.exceptions import ConnectionError as _ConnectionError, HTTPError as _HTTPError, RequestException as _RequestException, Timeout as _Timeout
            requests = _requests
            ConnectionError = _ConnectionError
            HTTPError = _HTTPError
            RequestException = _RequestException
            Timeout = _Timeout
        except Exception:  # pragma: no cover - optional dependency
            requests = _RequestsModulePlaceholder()
    return requests


# Singleton holder for DataFetcher instances
_FETCHER_SINGLETON: Any | None = None


def build_fetcher(config: Any):
    """Return a market data fetcher with safe fallbacks."""
    global _FETCHER_SINGLETON
    if _FETCHER_SINGLETON is not None:
        return _FETCHER_SINGLETON

    from ai_trading.alpaca_api import ALPACA_AVAILABLE
    bot_mod = importlib.import_module('ai_trading.core.bot_engine')
    DataFetcher = bot_mod.DataFetcher
    _ensure_http_client()
    yf_mod = _ensure_yfinance()
    req_mod = _ensure_requests()

    alpaca_ok = bool(os.getenv('ALPACA_API_KEY') and os.getenv('ALPACA_SECRET_KEY'))
    has_keys = alpaca_ok
    if ALPACA_AVAILABLE and has_keys:
        logger.info('DATA_FETCHER_BUILD', extra={'source': 'alpaca'})
        fetcher = DataFetcher()
        setattr(fetcher, 'source', 'alpaca')
        _FETCHER_SINGLETON = fetcher
        return fetcher
    if getattr(yf_mod, 'download', None) is not None and getattr(req_mod, 'get', None) is not None:
        logger.info('DATA_FETCHER_BUILD', extra={'source': 'yfinance'})
        fetcher = DataFetcher()
        setattr(fetcher, 'source', 'yfinance')
        _FETCHER_SINGLETON = fetcher
        return fetcher
    if getattr(req_mod, 'get', None) is not None:
        logger.warning('DATA_FETCHER_BUILD_FALLBACK', extra={'source': 'yahoo-requests'})
        fetcher = DataFetcher()
        setattr(fetcher, 'source', 'fallback')
        _FETCHER_SINGLETON = fetcher
        return fetcher
    logger.error('DATA_FETCHER_UNAVAILABLE', extra={'reason': 'no deps'})
    raise DataFetchError('No market data source available')

def _fetch_bars(symbol: str, start: Any, end: Any, timeframe: str, *, feed: str=_DEFAULT_FEED, adjustment: str='raw') -> pd.DataFrame:
    """Fetch bars from Alpaca v2 with alt-feed fallback."""
    pd = _ensure_pandas()
    _ensure_requests()
    if pd is None or getattr(requests, "get", None) is None:
        raise RuntimeError('requests not available')
    _start = ensure_datetime(start)
    _end = ensure_datetime(end)
    _interval = _canon_tf(timeframe)
    _feed = _canon_feed(feed or _DEFAULT_FEED)

    def _tags() -> dict[str, str]:
        return {'provider': 'alpaca', 'symbol': symbol, 'feed': _feed, 'timeframe': _interval}

    def _req(fallback: tuple[str, str, _dt.datetime, _dt.datetime] | None) -> pd.DataFrame:
        nonlocal _interval, _feed, _start, _end
        params = {'symbols': symbol, 'timeframe': _interval, 'start': _start.isoformat(), 'end': _end.isoformat(), 'limit': 10000, 'feed': _feed, 'adjustment': adjustment}
        if getattr(requests, "get", None) is None:
            raise RuntimeError('requests not available')
        url = 'https://data.alpaca.markets/v2/stocks/bars'
        headers = {'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY', ''), 'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY', '')}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            status = resp.status_code
            text = (resp.text or '').strip()
            ctype = (resp.headers.get('Content-Type') or '').lower()
        except Timeout as e:
            logger.warning('DATA_SOURCE_HTTP_ERROR', extra=_norm_extra({'provider': 'alpaca', 'feed': _feed, 'timeframe': _interval, 'error': str(e)}))
            _incr('data.fetch.timeout', value=1.0, tags=_tags())
            if fallback:
                _interval, _feed, _start, _end = fallback
                _incr('data.fetch.fallback_attempt', value=1.0, tags=_tags())
                payload = _format_fallback_payload_df(_interval, _feed, _start, _end)
                logger.info('DATA_SOURCE_FALLBACK_ATTEMPT', extra={'provider': 'alpaca', 'fallback': payload})
                return _req(None)
            raise
        except ConnectionError as e:
            logger.warning('DATA_SOURCE_HTTP_ERROR', extra=_norm_extra({'provider': 'alpaca', 'feed': _feed, 'timeframe': _interval, 'error': str(e)}))
            _incr('data.fetch.connection_error', value=1.0, tags=_tags())
            if fallback:
                _interval, _feed, _start, _end = fallback
                _incr('data.fetch.fallback_attempt', value=1.0, tags=_tags())
                payload = _format_fallback_payload_df(_interval, _feed, _start, _end)
                logger.info('DATA_SOURCE_FALLBACK_ATTEMPT', extra={'provider': 'alpaca', 'fallback': payload})
                return _req(None)
            raise
        except (HTTPError, RequestException, ValueError, KeyError) as e:
            logger.warning('DATA_SOURCE_HTTP_ERROR', extra=_norm_extra({'provider': 'alpaca', 'feed': _feed, 'timeframe': _interval, 'error': str(e)}))
            _incr('data.fetch.error', value=1.0, tags=_tags())
            if fallback:
                _interval, _feed, _start, _end = fallback
                _incr('data.fetch.fallback_attempt', value=1.0, tags=_tags())
                payload = _format_fallback_payload_df(_interval, _feed, _start, _end)
                logger.info('DATA_SOURCE_FALLBACK_ATTEMPT', extra={'provider': 'alpaca', 'fallback': payload})
                return _req(None)
            raise
        payload: dict[str, Any] | list[Any] = {}
        if status != 400 and text:
            if 'json' in ctype:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = {}
        data = []
        if isinstance(payload, dict):
            if 'bars' in payload and isinstance(payload['bars'], list):
                data = payload['bars']
            elif symbol in payload and isinstance(payload[symbol], dict) and ('bars' in payload[symbol]):
                data = payload[symbol]['bars']
        elif isinstance(payload, list):
            data = payload
        if status == 400:
            raise ValueError('Invalid feed or bad request')
        if status in (401, 403):
            _incr('data.fetch.unauthorized', value=1.0, tags=_tags())
            logger.warning(
                'UNAUTHORIZED_SIP' if _feed == 'sip' else 'DATA_SOURCE_UNAUTHORIZED',
                extra=_norm_extra({'provider': 'alpaca', 'status': 'unauthorized', 'feed': _feed, 'timeframe': _interval}),
            )
            if _feed == 'sip':
                return pd.DataFrame()
            if fallback:
                _interval, _feed, _start, _end = fallback
                _incr('data.fetch.fallback_attempt', value=1.0, tags=_tags())
                payload = _format_fallback_payload_df(_interval, _feed, _start, _end)
                logger.info('DATA_SOURCE_FALLBACK_ATTEMPT', extra={'provider': 'alpaca', 'fallback': payload})
                return _req(None)
            raise ValueError('unauthorized')
        if status == 429:
            _incr('data.fetch.rate_limited', value=1.0, tags=_tags())
            logger.warning('DATA_SOURCE_RATE_LIMITED', extra=_norm_extra({'provider': 'alpaca', 'status': 'rate_limited', 'feed': _feed, 'timeframe': _interval}))
            if fallback:
                _interval, _feed, _start, _end = fallback
                _incr('data.fetch.fallback_attempt', value=1.0, tags=_tags())
                payload = _format_fallback_payload_df(_interval, _feed, _start, _end)
                logger.info('DATA_SOURCE_FALLBACK_ATTEMPT', extra={'provider': 'alpaca', 'fallback': payload})
                return _req(None)
            raise ValueError('rate_limited')
        df = pd.DataFrame(data)
        if df.empty:
            if fallback:
                _incr('data.fetch.empty', value=1.0, tags=_tags())
            if _interval.lower() in {'1day', 'day', '1d'}:
                try:
                    mdf = _fetch_bars(symbol, _start, _end, '1Min', feed=_feed, adjustment=adjustment)
                except (ValueError, RuntimeError):
                    mdf = pd.DataFrame()
                if not mdf.empty:
                    try:
                        if 'timestamp' in mdf.columns:
                            mdf['timestamp'] = pd.to_datetime(mdf['timestamp'], utc=True)
                            mdf.set_index('timestamp', inplace=True)
                        from ai_trading.data.bars import _resample_minutes_to_daily as _resample_to_daily
                        rdf = _resample_to_daily(mdf)
                    except (ImportError, ValueError, TypeError, KeyError):
                        mdf = pd.DataFrame()
                    else:
                        if rdf is not None and (not rdf.empty):
                            return rdf
            _now = datetime.now(UTC)
            _key = (symbol, 'AVAILABLE', _now.date().isoformat(), _feed, _interval)
            if _empty_should_emit(_key, _now):
                lvl = _empty_classify(is_market_open=False)
                cnt = _empty_record(_key, _now)
                logger.log(lvl, 'EMPTY_DATA', extra=_norm_extra({'provider': 'alpaca', 'status': 'empty', 'feed': _feed, 'timeframe': _interval, 'occurrences': cnt}))
            if fallback:
                _interval, _feed, _start, _end = fallback
                _incr('data.fetch.fallback_attempt', value=1.0, tags=_tags())
                payload = _format_fallback_payload_df(_interval, _feed, _start, _end)
                logger.info('DATA_SOURCE_FALLBACK_ATTEMPT', extra={'provider': 'alpaca', 'fallback': payload})
                return _req(None)
            raise ValueError('empty_bars')
        ts_col = None
        for c in df.columns:
            if c.lower() in ('t', 'timestamp', 'time'):
                ts_col = c
                break
        if ts_col:
            df['timestamp'] = pd.to_datetime(df[ts_col], utc=True)
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime([], utc=True)
        rename = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df.set_index('timestamp', inplace=True, drop=False)
        _incr('data.fetch.success', value=1.0, tags=_tags())
        return df
    alt_feed = 'iex' if _feed != 'iex' else 'sip'
    fallback = (_interval, alt_feed, _start, _end)
    return _req(fallback)

def get_minute_df(symbol: str, start: Any, end: Any, feed: str | None=None) -> pd.DataFrame:
    """Minute bars fetch with provider fallback and downgraded errors.
    Also updates in-memory minute cache for freshness checks."""
    pd = _ensure_pandas()
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    use_finnhub = (
        os.getenv('ENABLE_FINNHUB', '1').lower() not in ('0', 'false')
        and os.getenv('FINNHUB_API_KEY')
        and fh_fetcher is not None
        and not getattr(fh_fetcher, 'is_stub', False)
    )
    if use_finnhub:
        try:
            df = fh_fetcher.fetch(symbol, start_dt, end_dt, resolution='1')
        except (FinnhubAPIException, ValueError, NotImplementedError) as e:
            logger.debug('FINNHUB_FETCH_FAILED', extra={'symbol': symbol, 'err': str(e)})
            df = None
    else:
        logger.debug('FINNHUB_DISABLED', extra={'symbol': symbol})
        df = None
    if df is None or getattr(df, 'empty', True):
        try:
            df = _fetch_bars(symbol, start_dt, end_dt, '1Min', feed=feed or _DEFAULT_FEED)
        except (ValueError, RuntimeError) as e:
            logger.warning('ALPACA_FETCH_FAILED', extra={'symbol': symbol, 'err': str(e)})
            df = None
    if df is None or getattr(df, 'empty', True):
        max_span = _dt.timedelta(days=8)
        total_span = end_dt - start_dt
        if total_span > max_span:
            logger.warning(
                'YF_1M_RANGE_SPLIT',
                extra={
                    'symbol': symbol,
                    'start': start_dt.isoformat(),
                    'end': end_dt.isoformat(),
                    'max_days': 8,
                },
            )
            dfs: list[pd.DataFrame] = []  # type: ignore[var-annotated]
            cur_start = start_dt
            while cur_start < end_dt:
                cur_end = min(cur_start + max_span, end_dt)
                dfs.append(_yahoo_get_bars(symbol, cur_start, cur_end, interval='1m'))
                cur_start = cur_end
            if pd is not None and dfs:
                df = pd.concat(dfs, ignore_index=True)
            elif dfs:
                df = dfs[0]
            else:
                df = pd.DataFrame() if pd is not None else []  # type: ignore[assignment]
        else:
            df = _yahoo_get_bars(symbol, start_dt, end_dt, interval='1m')
    try:
        if pd is not None and isinstance(df, pd.DataFrame) and (not df.empty):
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                last_ts = int(pd.Timestamp(df.index[-1]).tz_convert('UTC').timestamp())
            elif 'timestamp' in df.columns:
                last_ts = int(pd.Timestamp(df['timestamp'].iloc[-1]).tz_convert('UTC').timestamp())
            else:
                last_ts = None
            if last_ts is not None:
                set_cached_minute_timestamp(symbol, last_ts)
    except (ValueError, TypeError, KeyError, AttributeError):
        pass
    return _post_process(df)

def get_bars(symbol: str, timeframe: str, start: Any, end: Any, *, feed: str | None=None, adjustment: str | None=None) -> pd.DataFrame:
    """Compatibility wrapper delegating to _fetch_bars."""
    S = get_settings()
    feed = feed or S.alpaca_data_feed
    adjustment = adjustment or S.alpaca_adjustment
    return _fetch_bars(symbol, start, end, timeframe, feed=feed, adjustment=adjustment)

def get_bars_batch(symbols: list[str], timeframe: str, start: Any, end: Any, *, feed: str | None=None, adjustment: str | None=None) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple symbols via get_bars."""
    return {sym: get_bars(sym, timeframe, start, end, feed=feed, adjustment=adjustment) for sym in symbols}

def fetch_minute_yfinance(symbol: str, start_dt: _dt.datetime, end_dt: _dt.datetime) -> pd.DataFrame:
    """Explicit helper for tests and optional direct Yahoo minute fetch."""
    df = _yahoo_get_bars(symbol, start_dt, end_dt, interval='1m')
    return _post_process(df)

def is_market_open() -> bool:
    """Simplistic market-hours check used in tests."""
    return True

def _build_daily_url(symbol: str, start: datetime, end: datetime) -> str:
    start_s = int(start.timestamp())
    end_s = int(end.timestamp())
    return (
        'https://query1.finance.yahoo.com/v8/finance/chart/'
        f"{symbol}?period1={start_s}&period2={end_s}&interval=1d",
    )

__all__ = [
    '_DEFAULT_FEED',
    '_VALID_FEEDS',
    'ensure_datetime',
    '_yahoo_get_bars',
    '_fetch_bars',
    'get_bars',
    'get_bars_batch',
    'fetch_minute_yfinance',
    'is_market_open',
    'get_last_available_bar',
    'fh_fetcher',
    'get_minute_df',
    'build_fetcher',
    'DataFetchError',
    'DataFetchException',
    'FinnhubAPIException',
    'get_cached_minute_timestamp',
    'set_cached_minute_timestamp',
    'clear_cached_minute_timestamp',
    'age_cached_minute_timestamps',
    'last_minute_bar_age_seconds',
    '_build_daily_url',
]

