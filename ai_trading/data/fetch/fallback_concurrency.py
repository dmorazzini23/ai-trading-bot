# ai_trading/data/fetch/fallback_concurrency.py
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator

try:  # Prefer config-managed environment lookups when available.
    from ai_trading.config.management import get_env  # type: ignore
except Exception:  # pragma: no cover - fallback when config management unavailable

    def get_env(key: str, default: object | None = None, *, cast: object | None = None) -> object | None:  # type: ignore[override]
        return os.getenv(key, default)  # type: ignore[arg-type]


_ENV_PRIORITY: tuple[str, ...] = (
    "AI_TRADING_FALLBACK_CONCURRENCY",
    "AI_TRADING_HTTP_HOST_LIMIT",
    "AI_TRADING_HOST_LIMIT",
    "HTTP_MAX_PER_HOST",
    "HTTP_MAX_WORKERS",
    "AI_HTTP_HOST_LIMIT",
)
_DEFAULT_LIMIT = 3

_semaphore: threading.BoundedSemaphore | None = None
_fallback_limit_value: int | None = None
_fallback_inflight: int = 0
_fallback_peak: int = 0
_counter_lock = threading.Lock()


def _coerce_limit(value: object | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return max(1, int(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _resolve_limit_from_env() -> int:
    for key in _ENV_PRIORITY:
        candidate = None
        try:
            candidate = get_env(key, None)
        except Exception:
            candidate = None
        if candidate in (None, ""):
            candidate = os.getenv(key)
        limit = _coerce_limit(candidate)
        if limit is not None:
            return limit
    return _DEFAULT_LIMIT


def _ensure_semaphore() -> None:
    global _semaphore, _fallback_limit_value
    if _semaphore is not None:
        return
    limit = _resolve_limit_from_env()
    _semaphore = threading.BoundedSemaphore(limit)
    _fallback_limit_value = limit


def _rebuild_semaphore(new_limit: int) -> None:
    global _semaphore, _fallback_limit_value
    adjusted_limit = max(new_limit, max(_fallback_inflight, 1))
    semaphore = threading.BoundedSemaphore(adjusted_limit)
    permits_in_use = min(_fallback_inflight, adjusted_limit)
    for _ in range(permits_in_use):
        # Non-blocking acquire trims the available permits to match inflight usage.
        semaphore.acquire(blocking=False)
    _semaphore = semaphore
    _fallback_limit_value = adjusted_limit


def reload_fallback_limit() -> int:
    """Refresh the concurrency limit from environment settings."""

    limit = _resolve_limit_from_env()
    with _counter_lock:
        _rebuild_semaphore(limit)
    return int(_fallback_limit_value or limit)


def _acquire_slot() -> None:
    _ensure_semaphore()
    assert _semaphore is not None  # For type checkers
    _semaphore.acquire()
    global _fallback_inflight, _fallback_peak
    with _counter_lock:
        _fallback_inflight += 1
        if _fallback_inflight > _fallback_peak:
            _fallback_peak = _fallback_inflight


def _release_slot() -> None:
    global _fallback_inflight
    assert _semaphore is not None  # _ensure_semaphore guarantees initialization
    with _counter_lock:
        if _fallback_inflight > 0:
            _fallback_inflight -= 1
        else:  # Defensive clamp when release is called more than acquire.
            _fallback_inflight = 0
    _semaphore.release()


@contextmanager
def fallback_slot() -> Iterator[None]:
    """Context manager that limits concurrent fallback fetch attempts."""

    _acquire_slot()
    try:
        yield
    finally:
        _release_slot()


@contextmanager
def limit_concurrency() -> Iterator[None]:
    """Backwards-compatible alias for :func:`fallback_slot`."""

    with fallback_slot():
        yield


def get_peak_concurrency() -> int:
    """Return the highest number of concurrent fallback slots observed."""

    with _counter_lock:
        return _fallback_peak


def get_active_slots() -> int:
    """Return the number of in-flight fallback operations."""

    with _counter_lock:
        return _fallback_inflight


def get_configured_host_limit() -> int:
    """Return the currently configured fallback concurrency limit."""

    _ensure_semaphore()
    return int(_fallback_limit_value or _DEFAULT_LIMIT)


def reset_fallback_counters(reset_limit: bool = False) -> None:
    """Reset inflight counters and optionally rebuild the semaphore."""

    global _fallback_inflight, _fallback_peak
    with _counter_lock:
        _fallback_inflight = 0
        _fallback_peak = 0
        if reset_limit:
            limit = _resolve_limit_from_env()
            _rebuild_semaphore(limit)
        elif _semaphore is not None and _fallback_limit_value is not None:
            _rebuild_semaphore(_fallback_limit_value)


__all__ = [
    "fallback_slot",
    "limit_concurrency",
    "reload_fallback_limit",
    "reset_fallback_counters",
    "get_peak_concurrency",
    "get_active_slots",
    "get_configured_host_limit",
]
