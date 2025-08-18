from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from datetime import datetime
import os

import pandas as pd
import requests

from ai_trading.logging import get_logger

# AI-AGENT-REF: lightweight stubs for data fetch routines
FINNHUB_AVAILABLE = True
YFIN_AVAILABLE = True

__all__ = [
    "FINNHUB_AVAILABLE",
    "YFIN_AVAILABLE",
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
]

_log = get_logger(__name__)


class DataFetchException(Exception):
    """Error raised when market data retrieval fails."""


# ---- datetime helpers ----
def ensure_datetime(dt_or_str, *, tz: str | None = "UTC") -> datetime:
    """Return timezone-aware datetime in UTC."""  # AI-AGENT-REF
    if isinstance(dt_or_str, datetime):
        dt_obj = dt_or_str
    elif isinstance(dt_or_str, str):
        dt_obj = datetime.fromisoformat(dt_or_str.replace("Z", "+00:00"))
    elif isinstance(dt_or_str, (int, float)):
        dt_obj = datetime.fromtimestamp(dt_or_str, tz=dt.UTC)
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
def get_bars(symbol: str, timeframe: str, start, end, /, *, feed=None) -> pd.DataFrame:
    if start is not None:
        start = ensure_datetime(start)
    if end is not None:
        end = ensure_datetime(end)
    try:
        if feed and hasattr(feed, "get_bars"):
            return feed.get_bars(symbol, timeframe, start, end) or pd.DataFrame()
    except (requests.exceptions.RequestException, DataFetchException) as exc:
        _log.info(f"feed.get_bars error {exc.__class__.__name__}", exc_info=True)
    except AttributeError:
        pass
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return pd.DataFrame(columns=cols)


def get_bars_batch(
    symbols: Iterable[str], timeframe: str, start, end, /, *, feed=None
) -> dict[str, pd.DataFrame]:
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
    start = ensure_datetime(start) if start else None
    end = ensure_datetime(end) if end else None
    try:
        if feed and hasattr(feed, "get_bars"):
            return feed.get_bars(symbol, "1Min", start, end) or pd.DataFrame()
    except (requests.exceptions.RequestException, DataFetchException) as exc:
        _log.info(f"feed.get_bars error {exc.__class__.__name__}", exc_info=True)
    except AttributeError:
        pass
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return pd.DataFrame(columns=cols)


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


def _build_daily_url(symbol: str, start: datetime, end: datetime) -> str:
    """
    Construct a real Alpaca data URL for daily bars. Tests may monkeypatch this,
    but the production CLI benefits from a valid URL.
    """
    from ai_trading.settings import get_settings

    S = get_settings()
    base = getattr(S, "alpaca_data_base_url", "https://data.alpaca.markets")
    start_iso = ensure_datetime(start).isoformat().replace("+00:00", "Z")
    end_iso = ensure_datetime(end).isoformat().replace("+00:00", "Z")
    return (
        f"{base}/v2/stocks/{symbol}/bars"
        f"?timeframe=1Day&start={start_iso}&end={end_iso}&feed=iex"
    )
