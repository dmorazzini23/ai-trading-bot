from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from threading import RLock
from typing import Any

try:
    import finnhub  # noqa: F401

    FINNHUB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    finnhub = None  # type: ignore
    FINNHUB_AVAILABLE = False

__all__ = [
    "ensure_datetime",
    "get_minute_bars",
    "get_bars",  # legacy alias
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_cached_minute_timestamp",
    "age_cached_minute_timestamp",
    "retry",
    "FINNHUB_AVAILABLE",
]

# --- UTC normalizer ---------------------------------------------------------


def ensure_datetime(dt_like: Any) -> datetime:
    """
    Convert dt-like input (datetime|int|float|str) into timezone-aware UTC datetime.
    - naive datetime -> UTC
    - epoch seconds (int/float) -> UTC
    - RFC3339/ISO8601 str -> UTC
    """
    if isinstance(dt_like, datetime):
        return dt_like if dt_like.tzinfo else dt_like.replace(tzinfo=UTC)
    if isinstance(dt_like, int | float):
        return datetime.fromtimestamp(float(dt_like), tz=UTC)
    if isinstance(dt_like, str):
        # Loose parse without external deps
        try:
            # Support "YYYY-MM-DDTHH:MM:SS[.fff][Z]"
            ds = dt_like.rstrip("Z")
            # datetime.fromisoformat can't parse "Z"; we removed it above
            dt = datetime.fromisoformat(ds)
        except Exception:
            raise ValueError(f"Unsupported datetime string: {dt_like!r}") from None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt
    raise TypeError(f"Unsupported dt_like type: {type(dt_like)}")


# --- Simple retry utility ---------------------------------------------------


def retry(
    fn: Callable[[], Any],
    retries: int = 3,
    backoff_s: float = 0.25,
    retry_on: Iterable[type] | None = None,
) -> Any:
    import time

    attempts = 0
    retry_on = tuple(retry_on or (Exception,))
    while True:
        try:
            return fn()
        except retry_on:
            attempts += 1
            if attempts > retries:
                raise
            time.sleep(backoff_s * attempts)


# --- Thread-safe minute-level cache ----------------------------------------

_MINUTE_CACHE_LOCK = RLock()
_MINUTE_CACHE: dict[str, int] = {}


def get_cached_minute_timestamp(symbol: str) -> int | None:
    with _MINUTE_CACHE_LOCK:
        return _MINUTE_CACHE.get(symbol)


def set_cached_minute_timestamp(symbol: str, ts: int) -> None:
    if not isinstance(ts, int | float):
        raise TypeError("ts must be epoch seconds (int/float)")
    with _MINUTE_CACHE_LOCK:
        _MINUTE_CACHE[symbol] = int(ts)


def clear_cached_minute_timestamp(symbol: str) -> None:
    with _MINUTE_CACHE_LOCK:
        _MINUTE_CACHE.pop(symbol, None)


def age_cached_minute_timestamp(symbol: str, now_ts: int | None = None) -> int | None:
    with _MINUTE_CACHE_LOCK:
        ts = _MINUTE_CACHE.get(symbol)
    if ts is None:
        return None
    if now_ts is None:
        from time import time as _now

        now_ts = int(_now())
    return int(now_ts) - int(ts)


# --- Bars fetcher: minimal/patchable surface --------------------------------


def get_minute_bars(symbol: str, start: Any, end: Any, limit: int = 100) -> list[dict]:
    """
    Minimal bar fetcher. In production, this should query your data source.
    In tests, it's commonly monkeypatched. Returns list of dicts (time, open, high, low, close, volume).
    """
    _ = ensure_datetime(start), ensure_datetime(end)  # normalize only; no I/O here
    return []


# Legacy alias required by older code paths / bot_engine
get_bars = get_minute_bars
