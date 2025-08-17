from __future__ import annotations

import threading
import time
from collections.abc import Iterable  # noqa: F401
from datetime import UTC, datetime
from typing import Optional  # noqa: F401

try:
    import finnhub  # type: ignore

    FINNHUB_AVAILABLE = True
except Exception:  # noqa: BLE001
    finnhub = None  # type: ignore[assignment]
    FINNHUB_AVAILABLE = False

__all__ = [
    "ensure_datetime",
    "get_minute_df",
    "get_bars",  # legacy alias
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_minute_cache",
    "age_minute_cache",
    "retry",
    "FINNHUB_AVAILABLE",
]


# --- UTC normalizer ---
def ensure_datetime(value: datetime | str | int | float | None) -> datetime:
    """
    Normalize many datetime inputs into an aware UTC datetime.
    Accepts ISO strings, epoch seconds, naive/tz-aware datetimes,
    or None (returns utcnow()).
    """
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):  # noqa: UP038
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            dt = datetime.fromtimestamp(float(value), tz=UTC)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise TypeError(f"Unsupported datetime value: {type(value)!r}")


# --- minute-bar cache (thread-safe, import-safe) ---
_MINUTE_CACHE_LOCK = threading.RLock()
# symbol -> (last_minute_epoch_seconds, inserted_epoch_seconds)
_MINUTE_CACHE: dict[str, tuple[int, float]] = {}


def get_cached_minute_timestamp(symbol: str) -> int | None:
    with _MINUTE_CACHE_LOCK:
        tup = _MINUTE_CACHE.get(symbol.upper())
        return tup[0] if tup else None


def set_cached_minute_timestamp(symbol: str, ts: int) -> None:
    with _MINUTE_CACHE_LOCK:
        _MINUTE_CACHE[symbol.upper()] = (int(ts), time.time())


def clear_minute_cache(symbol: str | None = None) -> None:
    with _MINUTE_CACHE_LOCK:
        if symbol is None:
            _MINUTE_CACHE.clear()
        else:
            _MINUTE_CACHE.pop(symbol.upper(), None)


def age_minute_cache(max_age_seconds: int) -> int:
    cutoff = time.time() - int(max_age_seconds)
    removed = 0
    with _MINUTE_CACHE_LOCK:
        for k in list(_MINUTE_CACHE.keys()):
            _, inserted = _MINUTE_CACHE[k]
            if inserted < cutoff:
                _MINUTE_CACHE.pop(k, None)
                removed += 1
    return removed


# Backwards compat aliases
def clear_cached_minute_timestamp(
    symbol: str | None = None,
) -> None:  # pragma: no cover - legacy
    clear_minute_cache(symbol)


def age_cached_minute_timestamps(
    max_age_seconds: int,
) -> int:  # pragma: no cover - legacy
    return age_minute_cache(max_age_seconds)


# --- simple retry helper used by get_minute_df ---
def _retry(n: int, delay: float, fn, *args, **kwargs):
    last_err = None
    for _ in range(max(1, n)):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
    raise last_err  # type: ignore[misc]


retry = _retry  # AI-AGENT-REF: export retry helper


# --- bar fetcher (minimal, patchable) ---
def get_minute_df(
    symbol: str,
    start: datetime | str | int | float | None = None,
    end: datetime | str | int | float | None = None,
    retries: int = 2,
    delay: float = 0.25,
):
    """
    Minimal, patchable fetcher used by tests; actual provider injected elsewhere.
    Returns a pandas.DataFrame-like object in real runs; tests patch this.
    """
    s = ensure_datetime(start)
    e = ensure_datetime(end)
    return _retry(retries, delay, _fetch_bars_impl, symbol, s, e)


def _fetch_bars_impl(symbol: str, start: datetime, end: datetime):
    # Placeholder implementation; in production replaced/monkeypatched by tests.
    raise RuntimeError("No data provider configured")


# legacy alias required by older imports in bot_engine and runner
get_bars = get_minute_df
