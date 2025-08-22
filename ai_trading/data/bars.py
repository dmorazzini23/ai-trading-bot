from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ai_trading.config import get_settings
from ai_trading.data.market_calendar import previous_trading_session, rth_session_utc
from ai_trading.data_fetcher import get_bars, get_minute_df
from ai_trading.data_fetcher import (
    get_bars as http_get_bars,  # AI-AGENT-REF: fallback helpers
)
from ai_trading.logging import get_logger
from ai_trading.logging.empty_policy import (
    classify as _empty_classify,
)
from ai_trading.logging.empty_policy import (
    record as _empty_record,
)
from ai_trading.logging.empty_policy import (
    should_emit as _empty_should_emit,
)
from ai_trading.logging.normalize import (  # AI-AGENT-REF: central canonicalizers
    canon_feed as _canon_feed,
)
from ai_trading.logging.normalize import (
    canon_timeframe as _canon_tf,
)
from ai_trading.utils.time import now_utc

from .timeutils import ensure_utc_datetime, expected_regular_minutes

_log = get_logger(__name__)

"""AI-AGENT-REF: canonicalizers moved to ai_trading.logging.normalize"""
# AI-AGENT-REF: canonical fallback payload builder
def _format_fallback_payload(tf_str: str, feed_str: str, start_utc: datetime, end_utc: datetime) -> list[str]:
    s = start_utc.astimezone(UTC).isoformat()
    e = end_utc.astimezone(UTC).isoformat()
    return [tf_str, feed_str, s, e]


# AI-AGENT-REF: debug helper mapping ET window to UTC
def _log_fallback_window_debug(logger, day_et: date, start_utc: datetime, end_utc: datetime) -> None:
    try:
        logger.debug(
            "DATA_FALLBACK_WINDOW_DEBUG",
            extra={
                "et_day": day_et.isoformat(),
                "rth_et": "09:30-16:00",
                "rth_utc": f"{start_utc.astimezone(UTC).isoformat()}..{end_utc.astimezone(UTC).isoformat()}",
            },
        )
    except Exception:
        pass

# Light, local Alpaca shims so this module never needs bot_engine
try:
    from alpaca.data.requests import StockBarsRequest  # type: ignore
except Exception:  # pragma: no cover

    class StockBarsRequest:  # type: ignore
        pass


try:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore
except Exception:  # pragma: no cover

    class TimeFrame:  # type: ignore
        def __init__(self, n: int, unit: Any) -> None:
            self.n = n
            self.unit = unit

    class TimeFrameUnit:  # type: ignore
        Day = "Day"
        Minute = "Minute"


# Keep exception scope narrow and local
COMMON_EXC = (
    ValueError,
    KeyError,
    AttributeError,
    TypeError,
    RuntimeError,
    ImportError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _ensure_df(obj: Any) -> pd.DataFrame:
    """Best-effort conversion to DataFrame, never raises."""
    try:
        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.DataFrame):
            return obj
        if hasattr(obj, "df"):
            df = getattr(obj, "df", None)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return pd.DataFrame(obj) if obj is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def empty_bars_dataframe() -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    return pd.DataFrame(columns=cols)  # AI-AGENT-REF: shared empty DataFrame


def _create_empty_bars_dataframe() -> pd.DataFrame:
    return empty_bars_dataframe()


def _is_minute_timeframe(tf) -> bool:
    try:
        return str(tf).lower() in ("1min", "1m", "minute", "1 minute")
    except Exception:
        return False  # AI-AGENT-REF: broad but safe


def safe_get_stock_bars(
    client: Any, request: StockBarsRequest, symbol: str, context: str = ""
) -> pd.DataFrame:
    """
    Safely fetch stock bars via Alpaca client and always return a DataFrame.
    This is a faithful move of the original implementation from bot_engine,
    with identical behavior and logging fields.
    """
    now = now_utc()
    prev_open, _ = rth_session_utc(previous_trading_session(now.date()))  # AI-AGENT-REF
    end_dt = ensure_utc_datetime(
        getattr(request, "end", None) or now,
        default=now,
        clamp_to="eod",
        allow_callables=False,
    )
    start_dt = ensure_utc_datetime(
        getattr(request, "start", None) or prev_open,
        default=prev_open,
        clamp_to="bod",
        allow_callables=False,
    )
    try:
        try:
            response = client.get_stock_bars(request)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (401, 403):
                _log.error(
                    "ALPACA_BARS_UNAUTHORIZED",
                    extra={
                        "symbol": symbol,
                        "context": context,
                        "feed": _canon_feed(getattr(request, "feed", None)),
                    },
                )
                raise
            raise
        df = getattr(response, "df", None)
        if df is None or df.empty:
            _log.warning("ALPACA_BARS_EMPTY", extra={"symbol": symbol, "context": context})
            tf_str = _canon_tf(getattr(request, "timeframe", ""))
            feed_str = _canon_feed(getattr(request, "feed", None))
            if tf_str.lower() in {"1day", "day"}:
                mdf = get_minute_df(symbol, start_dt, end_dt, feed=feed_str)
                if mdf is not None and not mdf.empty:
                    rdf = _resample_minutes_to_daily(mdf)
                    if rdf is not None and not rdf.empty:
                        df = rdf
                    else:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()
                if df.empty:
                    try:
                        alt_req = StockBarsRequest(
                            symbol_or_symbols=symbol,
                            timeframe=TimeFrame.Day,
                            limit=2,
                            feed=feed_str,
                        )
                        alt_resp = client.get_stock_bars(alt_req)
                        df2 = getattr(alt_resp, "df", pd.DataFrame())
                        if isinstance(df2.index, pd.MultiIndex):
                            df2 = df2.xs(symbol, level=0, drop_level=False).droplevel(0)
                        df2 = df2.sort_index()
                        if not df2.empty:
                            last = df2.index[-1]
                            if last.date() == start_dt.date():
                                df = df2.loc[[last]]
                    except Exception as e:
                        status = getattr(e, "status_code", None)
                        if status in (401, 403):
                            _log.error(
                                "ALPACA_BARS_UNAUTHORIZED",
                                extra={"symbol": symbol, "context": context, "feed": feed_str},
                            )
                            raise
                        _log.warning(
                            "ALPACA_LIMIT_FETCH_FAILED",
                            extra={"symbol": symbol, "context": context, "error": str(e)},
                        )
            else:
                if _is_minute_timeframe(tf_str):
                    df = get_minute_df(symbol, start_dt, end_dt, feed=feed_str)
                else:
                    df = http_get_bars(symbol, tf_str, start_dt, end_dt, feed=feed_str)
            if df is None or df.empty:
                return _create_empty_bars_dataframe()

        # If MultiIndex (symbol, ts), select the symbol level
        if isinstance(df.index, pd.MultiIndex):
            try:
                df = df.xs(symbol, level=0, drop_level=False).droplevel(0)
            except (KeyError, ValueError):
                return _create_empty_bars_dataframe()

        # Normalize column casing if needed, keep original names if present
        # (match existing expectations in portfolio/core.py)
        if not df.empty:
            return df

        _now = datetime.now(UTC)
        _key = (
            symbol,
            str(context),
            _canon_feed(getattr(request, "feed", None)),  # AI-AGENT-REF
            _canon_tf(getattr(request, "timeframe", "")),
            _now.date().isoformat(),
        )
        if _empty_should_emit(_key, _now):
            lvl = _empty_classify(is_market_open=False)
            cnt = _empty_record(_key, _now)
            _log.log(
                lvl,
                "ALPACA_PARSE_EMPTY",
                extra={
                    "symbol": symbol,
                    "context": context,
                    "feed": _canon_feed(getattr(request, "feed", None)),
                    "timeframe": _canon_tf(getattr(request, "timeframe", "")),
                    "occurrences": cnt,
                },
            )
        return pd.DataFrame()
    except COMMON_EXC as e:
        _log.error(
            "ALPACA_BARS_FETCH_FAILED",
            extra={"symbol": symbol, "context": context, "error": str(e)},
        )
        if _is_minute_timeframe(getattr(request, "timeframe", "")):
            return _ensure_df(
                get_minute_df(
                    symbol,
                    start_dt,
                    end_dt,
                    feed=_canon_feed(getattr(request, "feed", None)),
                )
            )
        tf_str = _canon_tf(getattr(request, "timeframe", ""))
        feed_str = _canon_feed(getattr(request, "feed", None))
        df = http_get_bars(symbol, tf_str, start_dt, end_dt, feed=feed_str)
        return _ensure_df(df)  # AI-AGENT-REF: HTTP/Yahoo fallback on exception


def _fetch_daily_bars(client, symbol, start, end, **kwargs):
    start = ensure_utc_datetime(start)
    end = ensure_utc_datetime(end)
    get_bars_fn = getattr(client, "get_bars", None)
    if not callable(get_bars_fn):
        raise RuntimeError("Alpaca client missing get_bars()")
    try:
        return get_bars_fn(symbol, timeframe="1Day", start=start, end=end, **kwargs)
    except Exception as e:
        _log.exception("ALPACA_DAILY_FAILED", extra={"symbol": symbol, "error": str(e)})
        raise


def _get_minute_bars(symbol: str, start_dt: datetime, end_dt: datetime, feed: str) -> pd.DataFrame:
    try:
        df = get_bars(symbol=symbol, timeframe="1Min", start=start_dt, end=end_dt, feed=feed)
    except Exception:
        df = None
    if df is None or not hasattr(df, "empty") or getattr(df, "empty", True):
        return empty_bars_dataframe()
    return df


def _resample_minutes_to_daily(df, tz="America/New_York"):
    """Resample minute bars to daily OHLCV over regular trading hours."""  # AI-AGENT-REF
    if df is None or df.empty:
        return df
    try:
        mkt = df.copy()
        mkt = mkt.tz_convert(tz) if mkt.index.tz is not None else mkt.tz_localize(tz)
        mkt = mkt.between_time("09:30", "16:00", inclusive="both")
        o = mkt["open"].resample("1D").first()
        h = mkt["high"].resample("1D").max()
        l = mkt["low"].resample("1D").min()
        c = mkt["close"].resample("1D").last()
        v = mkt.get("volume")
        v = v.resample("1D").sum() if v is not None else None
        out = pd.concat({"open": o, "high": h, "low": l, "close": c}, axis=1)
        if v is not None:
            out["volume"] = v
        out = out.dropna(how="all").tz_convert("UTC")
        return out
    except Exception as e:
        _log.warning("RESAMPLE_DAILY_FAILED", extra={"error": str(e)})
        return df


def get_daily_bars(
    symbol: str,
    client,
    start: datetime,
    end: datetime,
    feed: str | None = None,
):
    """Fetch daily bars; fallback to alternate feed then resampled minutes."""  # AI-AGENT-REF
    S = get_settings()
    if feed is None:
        feed = S.alpaca_data_feed
    adjustment = S.alpaca_adjustment
    start = ensure_utc_datetime(start)
    end = ensure_utc_datetime(end)
    df = _fetch_daily_bars(client, symbol, start, end, feed=feed, adjustment=adjustment)
    if df is not None and not df.empty:
        return df
    alt = "iex" if feed == "sip" else "sip"
    df = _fetch_daily_bars(client, symbol, start, end, feed=alt, adjustment=adjustment)
    if df is not None and not df.empty:
        return df
    try:
        minutes_start = end - timedelta(days=5)
        mdf = _get_minute_bars(symbol, minutes_start, end, feed=feed)
        if mdf is not None and not mdf.empty:
            rdf = _resample_minutes_to_daily(mdf)
            if rdf is not None and not rdf.empty:
                _log.info(
                    "DAILY_FALLBACK_RESAMPLED",
                    extra={"symbol": symbol, "rows": len(rdf)},
                )
                return rdf
    except Exception as e:
        _log.warning(
            "DAILY_MINUTE_RESAMPLE_FAILED", extra={"symbol": symbol, "error": str(e)}
        )
    raise ValueError("empty_bars")


def _minute_fallback_window(now_utc: datetime) -> tuple[datetime, datetime]:
    """Compute NYSE session for the current or previous trading day."""  # AI-AGENT-REF

    today_ny = now_utc.astimezone(ZoneInfo("America/New_York")).date()
    start_u, end_u = rth_session_utc(today_ny)
    if now_utc < start_u or now_utc > end_u:
        prev_day = previous_trading_session(today_ny)
        start_u, end_u = rth_session_utc(prev_day)
    return start_u, end_u


def fetch_minute_fallback(client, symbol, now_utc: datetime) -> pd.DataFrame:
    now_utc = ensure_utc_datetime(now_utc)
    start_u, end_u = _minute_fallback_window(now_utc)
    day_et = start_u.astimezone(ZoneInfo("America/New_York")).date()
    _log_fallback_window_debug(_log, day_et, start_u, end_u)
    feed_str = "iex"
    df = _get_minute_bars(symbol, start_u, end_u, feed=feed_str)
    rows = len(df)
    if rows < 300:
        _log.warning(
            "DATA_HEALTH_MINUTE_INCOMPLETE",
            extra={
                "rows": rows,
                "expected": expected_regular_minutes(),
                "start": start_u.astimezone(UTC).isoformat(),
                "end": end_u.astimezone(UTC).isoformat(),
                "feed": feed_str,
            },
        )
        df_sip = _get_minute_bars(symbol, start_u, end_u, feed="sip")
        if len(df_sip) > rows:
            df = df_sip
            feed_str = "sip"
            rows = len(df)
    payload = _format_fallback_payload("1Min", feed_str, start_u, end_u)
    _log.info("DATA_FALLBACK_ATTEMPT", extra={"provider": "alpaca", "fallback": payload})
    if rows >= 300:
        _log.info("DATA_HEALTH: minute fallback ok", extra={"rows": rows})
    return df


def _parse_bars(payload: Any, symbol: str, tz: str) -> pd.DataFrame:
    if not payload:
        return empty_bars_dataframe()
    if isinstance(payload, dict):
        bars = payload.get("bars") or payload.get("data") or payload.get("results")
        if not bars:
            return empty_bars_dataframe()
        try:
            return _ensure_df(pd.DataFrame(bars))
        except Exception:
            return empty_bars_dataframe()
    if isinstance(payload, pd.DataFrame):
        return payload
    return empty_bars_dataframe()
