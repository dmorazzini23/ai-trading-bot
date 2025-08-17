"""Unified data fetcher API with patchable client and UTC helpers."""

from __future__ import annotations

import sys as _sys
import time
from datetime import UTC, datetime
from threading import RLock
from typing import Any

try:  # pragma: no cover - pandas optional
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# ---------------------------------------------------------------------------
# Minute-bar timestamp cache (thread-safe, import-safe)
# ---------------------------------------------------------------------------

_minute_cache_lock = RLock()
_minute_cache: dict[str, int] = {}


def set_cached_minute_timestamp(symbol: str, ts: int) -> None:
    """Store latest processed minute timestamp for ``symbol``."""

    with _minute_cache_lock:
        _minute_cache[symbol] = int(ts)


def get_cached_minute_timestamp(symbol: str) -> int | None:
    """Return cached minute timestamp for ``symbol`` if present."""

    with _minute_cache_lock:
        return _minute_cache.get(symbol)


def age_cached_minute_timestamp(symbol: str, delta: int) -> int | None:
    """Age the cached timestamp for ``symbol`` by ``delta`` seconds."""

    with _minute_cache_lock:
        if symbol in _minute_cache:
            _minute_cache[symbol] = int(_minute_cache[symbol]) + int(delta)
            return _minute_cache[symbol]
        return None


def clear_cached_minute_timestamp(symbol: str) -> None:
    """Remove cached timestamp for ``symbol`` if present."""

    with _minute_cache_lock:
        _minute_cache.pop(symbol, None)


# ---------------------------------------------------------------------------
# Client management and datetime helpers
# ---------------------------------------------------------------------------

_DATA_CLIENT: Any | None = None
_CACHE: dict[tuple[str, datetime, datetime], Any] = {}


def set_data_client(client: Any) -> None:
    """Set the global data client used for fetching bars."""

    global _DATA_CLIENT  # noqa: PLW0603
    _DATA_CLIENT = client


def ensure_datetime(dt: datetime | str | int | float) -> datetime:
    """Return a timezone-aware UTC ``datetime`` for ``dt``."""

    if isinstance(dt, datetime):
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    if isinstance(dt, int | float):
        return datetime.fromtimestamp(float(dt), tz=UTC)
    if isinstance(dt, str):
        try:
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            parsed = datetime.fromtimestamp(float(dt), tz=UTC)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise TypeError(f"Unsupported datetime type: {type(dt)!r}")


def rfc3339(dt: datetime | str | int | float) -> str:
    """Return an RFC3339 string in UTC for ``dt``."""

    return ensure_datetime(dt).isoformat().replace("+00:00", "Z")


def get_minute_df(
    symbol: str,
    start: datetime | str | int | float,
    end: datetime | str | int | float,
    *,
    client: Any | None = None,
    retries: int = 3,
    backoff_s: float = 0.5,
):
    """Fetch minute bars for ``symbol`` with bounded retries."""

    cl = client or _DATA_CLIENT
    if cl is None:
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
            df = cl.get_stock_bars(symbol, start=start_dt, end=end_dt)
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


def get_historical_df(
    symbol: str,
    start: datetime | str | int | float,
    end: datetime | str | int | float,
    *,
    client: Any | None = None,
    **kwargs: Any,
):
    """Backward compatible wrapper for :func:`get_minute_df`."""

    return get_minute_df(symbol, start, end, client=client, **kwargs)


# Backward-compatible aliases -------------------------------------------------


def get_bars(*args: Any, **kwargs: Any):
    """Legacy alias for :func:`get_minute_df`."""

    return get_minute_df(*args, **kwargs)


get_historical_data = get_historical_df


_sys.modules.setdefault("data_fetcher", _sys.modules[__name__])


__all__ = [
    "ensure_datetime",
    "get_minute_df",
    "get_bars",
    "get_historical_df",
    "set_cached_minute_timestamp",
    "get_cached_minute_timestamp",
    "age_cached_minute_timestamp",
    "clear_cached_minute_timestamp",
]
