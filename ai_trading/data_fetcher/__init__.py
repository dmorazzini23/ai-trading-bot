"""Unified data fetcher API with patchable client and UTC helpers."""

from __future__ import annotations

import sys as _sys
import time
from datetime import UTC, date, datetime
from threading import RLock
from typing import Any

try:  # pragma: no cover - pandas optional
    import pandas as pd  # type: ignore
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


def get_cached_minute_timestamp(symbol: str) -> datetime | None:
    """Return cached minute timestamp for ``symbol`` if present."""

    with _MINUTE_CACHE_LOCK:
        return _MINUTE_CACHE.get(symbol)


def set_cached_minute_timestamp(symbol: str, ts: datetime) -> None:
    """Store latest processed minute timestamp for ``symbol``."""

    with _MINUTE_CACHE_LOCK:
        _MINUTE_CACHE[symbol] = ensure_datetime(ts)


def clear_cached_minute_timestamp(symbol: str) -> None:
    """Remove cached timestamp for ``symbol`` if present."""

    with _MINUTE_CACHE_LOCK:
        _MINUTE_CACHE.pop(symbol, None)


def age_cached_minute_timestamp(symbol: str, *, max_age_s: int) -> None:
    """Delete cache entry if older than ``max_age_s`` seconds."""

    with _MINUTE_CACHE_LOCK:
        ts = _MINUTE_CACHE.get(symbol)
        if ts is None:
            return
        if (datetime.now(UTC) - ts).total_seconds() > max_age_s:
            _MINUTE_CACHE.pop(symbol, None)


# Backwards compatibility helpers -------------------------------------------


def clear_cached_minute_cache(symbol: str | None = None) -> None:
    """Legacy helper to clear cache for ``symbol`` or all."""

    with _MINUTE_CACHE_LOCK:
        if symbol is None:
            _MINUTE_CACHE.clear()
        else:
            _MINUTE_CACHE.pop(symbol, None)


def get_cached_age_seconds(symbol: str, now: datetime | None = None) -> float | None:
    """Legacy helper returning age of cached timestamp in seconds."""

    ts = get_cached_minute_timestamp(symbol)
    if ts is None:
        return None
    current = ensure_datetime(now or datetime.now(UTC))
    return (current - ts).total_seconds()


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def ensure_datetime(
    value: str | int | float | datetime | date,
    *,
    assume_tz: str = "UTC",
) -> datetime:
    """Return a timezone-aware UTC datetime for ``value``."""

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    elif isinstance(value, str):
        if not value:
            raise ValueError("Empty datetime string")
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:  # pragma: no cover - best effort
            try:
                dt = datetime.fromtimestamp(float(value))
            except Exception as exc2:  # pragma: no cover - best effort
                raise ValueError(f"Invalid datetime string: {value}") from exc2
    else:
        raise TypeError(f"Unsupported datetime type: {type(value)!r}")

    if dt.tzinfo is None:
        if assume_tz == "UTC":
            dt = dt.replace(tzinfo=UTC)
        else:  # pragma: no cover - rarely used
            try:
                from zoneinfo import ZoneInfo

                dt = dt.replace(tzinfo=ZoneInfo(assume_tz))
            except Exception:
                dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

_RETRY_HTTP_CODES = {408, 429, 500, 502, 503, 504}


def get_bars(
    symbol: str,
    start: datetime | str | int | float | date,
    end: datetime | str | int | float | date,
    timeframe: str = "1Min",
    *,
    client: Any | None = None,
    retries: int = 3,
    retry_backoff_s: float = 0.5,
) -> Any:
    """Fetch bars for ``symbol`` with basic retry logic."""

    cl = client or _DATA_CLIENT
    if cl is None:
        raise RuntimeError("DATA_CLIENT not configured")

    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return cl.get_stock_bars(
                symbol,
                start=start_dt,
                end=end_dt,
                timeframe=timeframe,
            )
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if isinstance(status, int) and status in _RETRY_HTTP_CODES and attempt < retries - 1:
                time.sleep(retry_backoff_s * (attempt + 1))
                last_exc = exc
                continue
            raise
    if last_exc is not None:  # pragma: no cover - all retries failed
        raise last_exc


# Backward-compatible aliases -------------------------------------------------


def get_minute_df(*args: Any, **kwargs: Any) -> Any:
    return get_bars(*args, **kwargs)


def get_historical_df(*args: Any, **kwargs: Any) -> Any:
    return get_bars(*args, **kwargs)


def get_historical_data(*args: Any, **kwargs: Any) -> Any:
    return get_bars(*args, **kwargs)


_sys.modules.setdefault("data_fetcher", _sys.modules[__name__])

__all__ = [
    "ensure_datetime",
    "get_bars",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_cached_minute_timestamp",
    "age_cached_minute_timestamp",
]
