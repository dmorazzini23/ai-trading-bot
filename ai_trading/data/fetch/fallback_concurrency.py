from __future__ import annotations

from contextlib import contextmanager
import os
import threading


DEFAULT_FALLBACK_LIMIT = 4
RESETTABLE_PEAK = True

_LOCK = threading.Lock()
_COND = threading.Condition(_LOCK)
_CURRENT = 0
_PEAK = 0

_ENV_KEYS = (
    "AI_TRADING_FALLBACK_MAX_CONCURRENCY",
    "FALLBACK_MAX_CONCURRENCY",
)


def _normalize_limit(value: int | str | None) -> int | None:
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return None
    if limit <= 0:
        return None
    return limit


def _env_limit() -> int:
    for key in _ENV_KEYS:
        candidate = _normalize_limit(os.getenv(key))
        if candidate is not None:
            return candidate
    return DEFAULT_FALLBACK_LIMIT


def _effective_limit(limit: int | None) -> int:
    normalized = _normalize_limit(limit)
    if normalized is not None:
        return normalized
    return max(_env_limit(), 1)


def current() -> int:
    with _COND:
        return _CURRENT


def peak() -> int:
    with _COND:
        return _PEAK


def current_concurrency() -> int:
    return current()


def peak_concurrency() -> int:
    return peak()


def reset_peak_simultaneous_workers() -> None:
    global _PEAK
    with _COND:
        _PEAK = 0


def reset_tracking_state() -> None:
    global _CURRENT, _PEAK
    with _COND:
        _CURRENT = 0
        _PEAK = 0
        _COND.notify_all()


@contextmanager
def run_with_concurrency(limit: int | None = None):
    global _CURRENT, _PEAK

    effective_limit = _effective_limit(limit)
    with _COND:
        while _CURRENT >= effective_limit:
            _COND.wait(timeout=0.01)
        _CURRENT += 1
        if _CURRENT > _PEAK:
            _PEAK = _CURRENT
    try:
        yield
    finally:
        with _COND:
            _CURRENT = max(0, _CURRENT - 1)
            _COND.notify_all()


@contextmanager
def fallback_slot(limit: int | None = None):
    with run_with_concurrency(limit=limit):
        yield


__all__ = [
    "fallback_slot",
    "run_with_concurrency",
    "reset_peak_simultaneous_workers",
    "reset_tracking_state",
    "current",
    "peak",
    "current_concurrency",
    "peak_concurrency",
    "DEFAULT_FALLBACK_LIMIT",
    "RESETTABLE_PEAK",
]
