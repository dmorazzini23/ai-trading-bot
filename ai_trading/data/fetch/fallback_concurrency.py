# ai_trading/data/fetch/fallback_concurrency.py
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator


def _sanitize_limit(value: int) -> int:
    """Clamp *value* to the minimum supported host concurrency."""

    if value < 1:
        return 1
    return value


def _initial_host_limit() -> int:
    raw = os.getenv("AI_HTTP_HOST_LIMIT", "6")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = 6
    return _sanitize_limit(value)


def _read_env_limit(default: int) -> int:
    raw = os.getenv("AI_HTTP_HOST_LIMIT")
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return _sanitize_limit(value)


_HOST_LIMIT: int = _initial_host_limit()
_peak_concurrency: int = 0
_active: int = 0
_pending_limit: int | None = None
_lock = threading.RLock()
_condition = threading.Condition(_lock)


def _apply_pending_limit_locked() -> None:
    global _pending_limit

    if _pending_limit is None:
        return
    pending = _pending_limit
    if _active <= pending:
        _pending_limit = None
        _condition.notify_all()


def begin_fallback_submission_wave() -> None:
    """Refresh the limiter configuration using the latest environment value."""

    global _HOST_LIMIT, _pending_limit

    limit = _read_env_limit(_HOST_LIMIT)
    if limit == _HOST_LIMIT:
        return
    with _condition:
        _HOST_LIMIT = limit
        if _active <= limit:
            _pending_limit = None
            _condition.notify_all()
        else:
            _pending_limit = limit
            _condition.notify_all()


def _maybe_refresh_limit_for_new_wave() -> None:
    with _lock:
        if _active != 0:
            return
    begin_fallback_submission_wave()


def _acquire_slot() -> None:
    global _active, _peak_concurrency

    with _condition:
        while _active >= _HOST_LIMIT:
            _condition.wait()
        _active += 1
        if _active > _peak_concurrency:
            _peak_concurrency = _active


def _release_slot() -> None:
    global _active

    with _condition:
        if _active > 0:
            _active -= 1
        else:
            _active = 0
        _apply_pending_limit_locked()
        _condition.notify_all()


@contextmanager
def limit_concurrency() -> Iterator[None]:
    _maybe_refresh_limit_for_new_wave()
    _acquire_slot()
    try:
        yield
    finally:
        _release_slot()


@contextmanager
def fallback_slot() -> Iterator[None]:
    with limit_concurrency():
        yield


def get_peak_concurrency() -> int:
    return _peak_concurrency


def get_active_slots() -> int:
    with _lock:
        return _active


def get_configured_host_limit() -> int:
    return _HOST_LIMIT


__all__ = [
    "limit_concurrency",
    "fallback_slot",
    "begin_fallback_submission_wave",
    "get_peak_concurrency",
    "get_active_slots",
    "get_configured_host_limit",
]
