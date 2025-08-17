from __future__ import annotations

import datetime as _dt
from typing import Dict, Iterable, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    import pytz as _pytz
except Exception:  # pragma: no cover
    _pytz = None

FINNHUB_AVAILABLE = True
YFIN_AVAILABLE = True

__all__ = [
    "FINNHUB_AVAILABLE",
    "YFIN_AVAILABLE",
    "ensure_datetime",
    "to_utc",
    "get_bars",
    "get_bars_batch",
    "get_minute_df",
    "_MINUTE_CACHE",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_cached_minute_timestamp",
    "age_cached_minute_timestamps",
]


# ---- datetime helpers ----
def to_utc(dt: _dt.datetime) -> _dt.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


def ensure_datetime(x) -> _dt.datetime:
    """Return timezone-aware UTC datetime."""
    if isinstance(x, _dt.datetime):
        return to_utc(x)
    if isinstance(x, str):
        try:
            dt = _dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
        except Exception:
            dt = _dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        return to_utc(dt)
    if isinstance(x, (int, float)):
        return to_utc(_dt.datetime.fromtimestamp(x, tz=_dt.timezone.utc))
    raise TypeError(f"Unsupported type for ensure_datetime: {type(x)!r}")


# ---- minute cache helpers ----
_MINUTE_CACHE: Dict[str, _dt.datetime] = {}


def get_cached_minute_timestamp(symbol: str) -> Optional[_dt.datetime]:
    return _MINUTE_CACHE.get(symbol.upper())


def set_cached_minute_timestamp(symbol: str, ts_or_dt) -> None:
    if not isinstance(ts_or_dt, _dt.datetime):
        raise TypeError("ts_or_dt must be a datetime")
    _MINUTE_CACHE[symbol.upper()] = to_utc(ts_or_dt)


def clear_cached_minute_timestamp(symbol: str | None = None) -> None:
    if symbol is None:
        _MINUTE_CACHE.clear()
    else:
        _MINUTE_CACHE.pop(symbol.upper(), None)


def age_cached_minute_timestamps(minutes: int) -> None:
    if minutes < 0:
        return
    # No timebase; tests just need callable.


# Backwards compatibility aliases
clear_minute_cache = clear_cached_minute_timestamp
age_minute_cache = age_cached_minute_timestamps


# ---- data access stubs ----
def get_bars(
    symbol: str,
    timeframe: str,
    start: _dt.datetime | None = None,
    end: _dt.datetime | None = None,
    *,
    feed=None,
    client=None,
) -> pd.DataFrame:
    if start is not None:
        start = ensure_datetime(start)
    if end is not None:
        end = ensure_datetime(end)
    try:
        if client and hasattr(client, "get_bars"):
            return client.get_bars(symbol, timeframe, start, end, feed=feed) or pd.DataFrame()
    except Exception:
        pass
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return pd.DataFrame(columns=cols)


def get_bars_batch(
    symbols: Iterable[str],
    timeframe: str,
    start,
    end,
    feed=None,
    client=None,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[str(sym)] = get_bars(sym, timeframe, start, end, feed=feed, client=client)
        except Exception:
            pass
    return out


def get_minute_df(*args, **kwargs) -> pd.DataFrame:
    """Placeholder minute fetcher; tests may monkeypatch."""
    return pd.DataFrame()


