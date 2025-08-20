from __future__ import annotations

import datetime as dt
import os
import warnings
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from urllib.parse import urlencode

import pandas as pd
import requests

from ai_trading.alpaca_api import get_bars_df  # AI-AGENT-REF: canonical fetcher
from ai_trading.exc import COMMON_EXC, TRANSIENT_HTTP_EXC  # AI-AGENT-REF: Stage 2.1
from ai_trading.logging import get_logger
from ai_trading.utils.retry import retry_call  # AI-AGENT-REF: Stage 2.1

try:  # AI-AGENT-REF: optional import
    from alpaca_trade_api.rest import TimeFrame
except Exception:  # pragma: no cover  # noqa: BLE001
    TimeFrame = None  # type: ignore

warnings.warn(
    "ai_trading.data_fetcher: legacy helpers will change in v1.0",
    DeprecationWarning,
)  # AI-AGENT-REF: emit deprecation warning

# AI-AGENT-REF: lightweight stubs for data fetch routines
FINNHUB_AVAILABLE = True
YFIN_AVAILABLE = True
_DEFAULT_FEED = "iex"  # AI-AGENT-REF: default feed constant


class _TFUnit(str, Enum):  # AI-AGENT-REF: include Week unit
    Minute = "Minute"
    Hour = "Hour"
    Day = "Day"
    Week = "Week"

__all__ = [
    "FINNHUB_AVAILABLE",
    "YFIN_AVAILABLE",
    "_DEFAULT_FEED",
    "ensure_datetime",
    "get_bars",
    "get_bars_batch",
    "get_minute_df",
    "get_minute_bars_batch",
    "_MINUTE_CACHE",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_cached_minute_timestamp",
    "age_cached_minute_timestamps",
    "last_minute_bar_age_seconds",
    "DataFetchException",
    "DataFetchError",
    "_TFUnit",
    "_yahoo_get_bars",
    "requests",
    "get_bars_df",
]

_log = get_logger(__name__)


class DataFetchException(Exception):
    """Error raised when market data retrieval fails."""


class DataFetchError(DataFetchException):
    """Alias error used in tests."""  # AI-AGENT-REF


def _empty_bars_df() -> pd.DataFrame:
    """Return an empty OHLCV DataFrame with UTC timestamps."""
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).tz_localize("UTC")
    return df


def normalize_symbol_for_provider(symbol: str, provider: str) -> str:
    """Return symbol normalized for the data provider."""
    if provider.lower() == "yahoo":
        return symbol.replace(".", "-")
    return symbol


# ---- datetime helpers ----
def ensure_datetime(dt_or_str, *, tz: str | None = "UTC") -> datetime:
    """Return timezone-aware datetime in UTC."""  # AI-AGENT-REF
    if dt_or_str is None:
        raise ValueError("value_none")
    if isinstance(dt_or_str, pd.Timestamp):
        dt_obj = dt_or_str.to_pydatetime()
    elif isinstance(dt_or_str, datetime):
        dt_obj = dt_or_str
    elif isinstance(dt_or_str, str):
        if not dt_or_str:
            raise ValueError("empty_string")
        try:
            dt_obj = datetime.fromisoformat(dt_or_str.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError("invalid_string") from e
    else:
        raise TypeError(f"Unsupported type for ensure_datetime: {type(dt_or_str)!r}")
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.UTC)
    return dt_obj.astimezone(dt.UTC)


# ---- minute cache helpers ----
# AI-AGENT-REF: bounded minute cache
_MINUTE_CACHE: dict[str, int] = {}
_GLOBAL_MINUTE_TS: int | None = None
_MINUTE_CACHE_MAX = int(os.getenv("MINUTE_CACHE_MAX", "5000"))


def set_cached_minute_timestamp(symbol: str | None, ts_or_dt) -> None:
    """Store timestamp (epoch seconds) for ``symbol`` and global cache."""  # AI-AGENT-REF
    global _GLOBAL_MINUTE_TS
    if isinstance(ts_or_dt, datetime):
        ts = int(ensure_datetime(ts_or_dt).timestamp())
    else:
        ts = int(ts_or_dt)
    _GLOBAL_MINUTE_TS = ts
    if symbol:
        _MINUTE_CACHE[symbol.upper()] = ts
        # Size-based eviction (drop oldest ~20% on overflow)
        if len(_MINUTE_CACHE) > _MINUTE_CACHE_MAX:
            drop = max(1, _MINUTE_CACHE_MAX // 5)
            for k in list(_MINUTE_CACHE.keys())[:drop]:
                _MINUTE_CACHE.pop(k, None)


def get_cached_minute_timestamp(symbol: str | None = None) -> int | None:
    """Return cached timestamp for ``symbol`` or global cache."""  # AI-AGENT-REF
    if symbol is None:
        return _GLOBAL_MINUTE_TS
    return _MINUTE_CACHE.get(symbol.upper())


def clear_cached_minute_timestamp(symbol: str | None = None) -> None:
    if symbol is None:
        _MINUTE_CACHE.clear()
        global _GLOBAL_MINUTE_TS
        _GLOBAL_MINUTE_TS = None
    else:
        _MINUTE_CACHE.pop(symbol.upper(), None)


def age_cached_minute_timestamps(max_age_seconds: int) -> int:
    """Remove cached timestamps older than ``max_age_seconds``."""  # AI-AGENT-REF
    now = int(datetime.now(dt.UTC).timestamp())
    removed = 0
    for sym, ts in list(_MINUTE_CACHE.items()):
        if now - ts > max_age_seconds:
            _MINUTE_CACHE.pop(sym, None)
            removed += 1
    global _GLOBAL_MINUTE_TS
    if _GLOBAL_MINUTE_TS is not None and now - _GLOBAL_MINUTE_TS > max_age_seconds:
        _GLOBAL_MINUTE_TS = None
    return removed


def last_minute_bar_age_seconds(now: datetime | None = None) -> int:
    """Age of the most recent cached minute bar."""  # AI-AGENT-REF
    if _GLOBAL_MINUTE_TS is None:
        return 0
    current = ensure_datetime(now or datetime.now(dt.UTC))
    return int(current.timestamp() - _GLOBAL_MINUTE_TS)


# ---- data access stubs ----


def _alpaca_get_bars(
    client, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str = "1Day"
) -> pd.DataFrame:
    """Fetch bars via new Alpaca REST helper."""  # AI-AGENT-REF
    del client, start, end
    tf = TimeFrame.Day if timeframe.lower() in {"1d", "1day"} else TimeFrame.Minute
    return get_bars_df(symbol, tf)


def _to_utc_dt(dt_like):
    """Return a UTC-aware ``datetime.datetime`` or ``None``."""  # AI-AGENT-REF: robust tz
    if dt_like is None:
        return None
    if isinstance(dt_like, datetime):
        return (
            dt_like.replace(tzinfo=dt.UTC)
            if dt_like.tzinfo is None
            else dt_like.astimezone(dt.UTC)
        )
    try:
        ts = pd.Timestamp(dt_like)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except Exception:  # noqa: BLE001
        return ensure_datetime(dt_like)


def _yahoo_get_bars(
    symbol: str,
    start: datetime | pd.Timestamp | None,
    end: datetime | pd.Timestamp | None,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    """Fetch bars from Yahoo Finance if available."""
    try:
        import yfinance as yf  # type: ignore
        # AI-AGENT-REF: route yfinance TzCache to writable dir to silence warnings
        try:
            import tempfile, os as _os
            _tz_dir = _os.getenv("YFINANCE_TZCACHE") or _os.path.join(tempfile.gettempdir(), "py-yfinance")
            if hasattr(yf, "set_tz_cache_location"):
                yf.set_tz_cache_location(_tz_dir)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        return _empty_bars_df()

    interval = "1d" if timeframe.lower() in {"1d", "1day"} else "1m"
    sym = normalize_symbol_for_provider(symbol, "yahoo")
    start = _to_utc_dt(start)
    end = _to_utc_dt(end)
    try:
        df = yf.download(
            sym,
            start=start,
            end=end,
            interval=interval,
            progress=False,
        )
    except Exception:  # noqa: BLE001
        return _empty_bars_df()
    if df is None or df.empty:
        return _empty_bars_df()
    df = df.reset_index()
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA
    return df[keep].sort_values("timestamp")
def get_bars(symbol: str, timeframe: str, start, end, /, *, feed=None, client=None) -> pd.DataFrame:
    del client
    if start is not None:
        start = ensure_datetime(start)
    if end is not None:
        end = ensure_datetime(end)
    SAFE_EXC = COMMON_EXC + (DataFetchException,)
    if feed is not None:
        try:
            df = retry_call(
                lambda: _alpaca_get_bars(feed, symbol, start, end, timeframe),
                exceptions=TRANSIENT_HTTP_EXC,
            )
            if df is not None and not df.empty:
                return df
        except SAFE_EXC as exc:  # AI-AGENT-REF: Stage 2.1 narrow catch
            _log.info(f"feed.get_bars error {exc.__class__.__name__}", exc_info=True)
        except AttributeError:
            pass
    return _yahoo_get_bars(symbol, start, end, timeframe)


def get_bars_batch(
    symbols: Iterable[str], timeframe: str, start, end, /, *, feed=None, client=None
) -> dict[str, pd.DataFrame]:
    del client
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[str(sym)] = get_bars(sym, timeframe, start, end, feed=feed)
    return out


def get_minute_df(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    feed=None,
) -> pd.DataFrame:
    del start, end, feed
    try:
        df = get_bars_df(symbol, TimeFrame.Minute)
        return df
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "ALPACA_MINUTE_FAIL",
            extra={"symbol": symbol, "error": str(exc)},
        )
        raise


def get_minute_bars_batch(
    symbols: Iterable[str],
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    feed=None,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[str(sym)] = get_minute_df(sym, start, end, feed=feed)
    return out


def _to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_daily_url(symbol: str, start: datetime, end: datetime) -> str:
    base = "https://data.alpaca.markets/v2/stocks"
    params = {
        "start": _to_iso_z(ensure_datetime(start)),
        "end": _to_iso_z(ensure_datetime(end)),
        "timeframe": "1Day",
        "feed": "iex",
    }
    return f"{base}/{symbol}/bars?{urlencode(params)}"
