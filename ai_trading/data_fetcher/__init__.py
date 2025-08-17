"""Unified data fetcher API with patchable client and UTC helpers."""

from __future__ import annotations

import sys as _sys
import time
from datetime import UTC, date, datetime
from threading import RLock
from typing import Any

try:  # pragma: no cover - pandas optional
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# ---------------------------------------------------------------------------
# Global data client (patchable for tests)
# ---------------------------------------------------------------------------

_DATA_CLIENT: Any | None = None


def set_data_client(client: Any) -> None:
    """Set the global data client used for fetching bars."""

    global _DATA_CLIENT  # noqa: PLW0603
    _DATA_CLIENT = client


# ---------------------------------------------------------------------------
# Minute-bar timestamp cache (thread-safe, import-safe)
# ---------------------------------------------------------------------------

_MINUTE_CACHE: dict[str, datetime] = {}
_MINUTE_CACHE_LOCK = RLock()


def set_cached_minute_timestamp(symbol: str, ts: datetime) -> None:
    """Store latest processed minute timestamp for ``symbol``."""

    dt = ensure_datetime(ts)
    with _MINUTE_CACHE_LOCK:
        _MINUTE_CACHE[symbol] = dt


def get_cached_minute_timestamp(symbol: str) -> datetime | None:
    """Return cached minute timestamp for ``symbol`` if present."""

    with _MINUTE_CACHE_LOCK:
        return _MINUTE_CACHE.get(symbol)


def clear_cached_minute_cache(symbol: str | None = None) -> None:
    """Clear cache for ``symbol`` or all symbols if ``None``."""

    with _MINUTE_CACHE_LOCK:
        if symbol is None:
            _MINUTE_CACHE.clear()
        else:
            _MINUTE_CACHE.pop(symbol, None)


def get_cached_age_seconds(symbol: str, now: datetime | None = None) -> float | None:
    """Return age of cached timestamp in seconds for ``symbol``."""

    with _MINUTE_CACHE_LOCK:
        ts = _MINUTE_CACHE.get(symbol)
    if ts is None:
        return None
    current = ensure_datetime(now or datetime.now(UTC))
    return (current - ts).total_seconds()


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def ensure_datetime(value: str | int | float | datetime | date) -> datetime:
    """Return a timezone-aware UTC datetime for ``value``."""

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        if not value:
            raise ValueError("Empty datetime string")
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:  # pragma: no cover - best effort
            try:
                dt = datetime.fromtimestamp(float(value), tz=UTC)
            except Exception as exc2:  # pragma: no cover - best effort
                raise ValueError(f"Invalid datetime string: {value}") from exc2
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise TypeError(f"Unsupported datetime type: {type(value)!r}")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

_CACHE: dict[tuple[str, datetime, datetime], Any] = {}


def get_minute_df(
    *,
    symbol: str,
    start: datetime | str | int | float | date,
    end: datetime | str | int | float | date,
    retries: int = 3,
    backoff_s: float = 0.25,
    client: Any | None = None,
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
            time.sleep(backoff_s * (attempt + 1))
    raise last_exc  # pragma: no cover


# Backward-compatible aliases -------------------------------------------------


def get_bars(*args: Any, **kwargs: Any):
    """Legacy alias for :func:`get_minute_df`."""

    return get_minute_df(*args, **kwargs)


get_historical_df = get_minute_df
get_historical_data = get_minute_df

_sys.modules.setdefault("data_fetcher", _sys.modules[__name__])


__all__ = [
    "ensure_datetime",
    "get_minute_df",
    "get_bars",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_cached_minute_cache",
    "get_cached_age_seconds",
    "_DATA_CLIENT",
    "set_data_client",
]
