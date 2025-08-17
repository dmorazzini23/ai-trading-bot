"""Unified data fetcher API with patchable client and UTC helpers."""

from __future__ import annotations

import sys as _sys
import time
from datetime import UTC, datetime
from threading import RLock  # AI-AGENT-REF: thread-safe minute cache helpers
from typing import Any  # AI-AGENT-REF: explicit typing for cache

try:  # pragma: no cover - pandas optional
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# ======================================================================================
# Public minute-level cache helpers (required by tests/test_minute_cache_helpers.py)
# Thread-safe, no mutable default args, import-safe singletons.
# ======================================================================================

_MINUTE_CACHE_LOCK: RLock = RLock()  # AI-AGENT-REF: protect minute cache access
_MINUTE_CACHE: dict[str, Any] = {}  # AI-AGENT-REF: symbol -> (df, ts) or ts


def set_cached_minute_timestamp(symbol: str, ts: int) -> None:
    """Store latest processed minute timestamp for ``symbol``."""
    with _MINUTE_CACHE_LOCK:
        _MINUTE_CACHE[symbol] = int(ts)  # AI-AGENT-REF: store epoch seconds


def get_cached_minute_timestamp(symbol: str) -> pd.Timestamp | None:
    """Return cached minute timestamp for ``symbol`` if present."""
    with _MINUTE_CACHE_LOCK:
        entry = _MINUTE_CACHE.get(symbol)
    if entry is None:
        return None
    ts = entry[1] if isinstance(entry, tuple) else entry
    try:
        if pd is not None:
            ts = pd.Timestamp(ts)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            return ts
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def clear_minute_cache(symbol: str | None = None) -> None:
    """Clear cache for ``symbol`` or all symbols when ``symbol`` is ``None``."""
    with _MINUTE_CACHE_LOCK:
        if symbol is None:
            _MINUTE_CACHE.clear()  # AI-AGENT-REF: drop entire cache
        else:
            _MINUTE_CACHE.pop(symbol, None)  # AI-AGENT-REF: best-effort removal


def last_minute_bar_age_seconds(symbol: str) -> int | None:
    """Return age in seconds of latest cached minute bar; ``None`` if unknown."""
    ts = get_cached_minute_timestamp(symbol)
    if ts is None:
        return None
    now = pd.Timestamp.now(tz="UTC") if pd is not None else datetime.now(UTC)
    try:
        return int((now - ts).total_seconds())
    except Exception:  # pragma: no cover - defensive
        return None


_DATA_CLIENT: Any | None = None
_CACHE: dict[tuple[str, datetime, datetime], pd.DataFrame] = {}


def set_data_client(client: Any) -> None:
    """Set the global data client used for fetching bars."""
    global _DATA_CLIENT  # noqa: PLW0603
    _DATA_CLIENT = client


def ensure_datetime(dt: datetime | str) -> datetime:
    """Return a timezone-aware UTC ``datetime`` for ``dt``."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    elif hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()  # type: ignore[assignment]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def rfc3339(dt: datetime | str) -> str:
    """Return an RFC3339 string in UTC for ``dt``."""
    return ensure_datetime(dt).isoformat().replace("+00:00", "Z")


def get_minute_df(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    *,
    retries: int = 3,
    backoff_s: float = 0.5,
) -> pd.DataFrame:
    """Fetch minute bars for ``symbol`` with bounded retries."""
    if _DATA_CLIENT is None:
        raise RuntimeError("DATA_CLIENT not configured")
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    key = (symbol, start_dt, end_dt)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            df = _DATA_CLIENT.get_stock_bars(symbol, start=start_dt, end=end_dt)
            _CACHE[key] = df
            return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            if (
                "Invalid format for parameter start" not in msg and "error parsing" not in msg
            ) or attempt >= retries - 1:
                raise
            time.sleep(backoff_s * (2**attempt))
    raise last_exc  # pragma: no cover - unreachable


def get_historical_data(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Backward compatible wrapper for :func:`get_minute_df`."""
    return get_minute_df(symbol, start, end, **kwargs)


get_bars = get_minute_df

_sys.modules.setdefault("data_fetcher", _sys.modules[__name__])


__all__ = [
    "ensure_datetime",
    "rfc3339",
    "get_bars",
    "get_minute_df",
    "get_historical_data",
    "set_data_client",
    "_DATA_CLIENT",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_minute_cache",
    "last_minute_bar_age_seconds",
]
