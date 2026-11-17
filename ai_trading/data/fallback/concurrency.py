"""Async helpers for running fallback tasks with bounded concurrency.

The closure-rebinding logic below walks dataclasses, ``SimpleNamespace``
instances, standard containers, and slot-based objects to recreate any
foreign-loop asyncio locks on the currently running loop. If additional
container types start capturing asyncio primitives, keep ``_scan`` in sync
with those structures.
"""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import asyncio
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from types import ModuleType, SimpleNamespace
from typing import Any, TypeVar, cast

from ai_trading.config.management import get_env
from ai_trading.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional during stubbed tests
    from ai_trading.net import http_host_limit as _http_host_limit
except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - fallback when module unavailable
    _http_host_limit = None

try:  # pragma: no cover - optional during stubbed tests
    from ai_trading.http import pooling as _http_pooling
except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - pooling optional during stubbed tests
    _pooling_host_limit = None
    _pooling_get_host_semaphore = None
    _pooling_reload_host_limit = None
    _pooling_get_limit_snapshot = None
    _pooling_refresh_host_semaphore = None
    _pooling_record_concurrency = None
else:  # pragma: no cover - exercised in integration tests
    _pooling_host_limit = getattr(_http_pooling, "get_host_limit", None)
    _pooling_get_host_semaphore = getattr(_http_pooling, "get_host_semaphore", None)
    _pooling_reload_host_limit = getattr(
        _http_pooling, "reload_host_limit_if_env_changed", None,
    )
    _pooling_get_limit_snapshot = getattr(
        _http_pooling, "get_host_limit_snapshot", None,
    )
    _pooling_refresh_host_semaphore = getattr(
        _http_pooling, "refresh_host_semaphore", None,
    )
    _pooling_record_concurrency = getattr(_http_pooling, "record_concurrency", None)

try:  # pragma: no cover - optional dependency; only present under tests
    import freezegun.api as _freezegun_api
except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - freezegun not installed in production
    _freezegun_api = None


_POOLING_LIMIT_STATE: tuple[int, int] | None = None
_LOCAL_POOLING_VERSION: int = 0


def _record_pooling_snapshot(limit: int, version: int) -> None:
    """Persist a normalised pooling snapshot and track the latest version."""
    global _POOLING_LIMIT_STATE, _LOCAL_POOLING_VERSION

    _POOLING_LIMIT_STATE = (limit, version)
    _LOCAL_POOLING_VERSION = max(_LOCAL_POOLING_VERSION, version)


def _invalidate_pooling_snapshot() -> None:
    """Force the next limit lookup to recompute the pooling snapshot."""
    global _POOLING_LIMIT_STATE

    _POOLING_LIMIT_STATE = None


def _normalise_pooling_state(snapshot: object | None) -> tuple[int, int] | None:
    """Return ``(limit, version)`` when ``snapshot`` exposes that metadata."""
    if snapshot is None:
        return None

    limit = getattr(snapshot, "limit", None)
    version = getattr(snapshot, "version", None)
    if not isinstance(limit, int) or not isinstance(version, int):
        if isinstance(snapshot, tuple) and len(snapshot) >= 2:
            limit, version = snapshot[0], snapshot[1]
        else:
            return None
    try:
        limit = int(limit)
        version = int(version)
    except (TypeError, ValueError):
        return None
    limit = max(limit, 1)
    return limit, version


T = TypeVar("T")

_ASYNCIO_PRIMITIVE_NAMES = {"Lock", "Semaphore", "Event", "Condition", "BoundedSemaphore"}
_ASYNCIO_LOCK_NAMES = {"Lock", "Semaphore", "BoundedSemaphore"}


def _collect_asyncio_primitive_types() -> tuple[type, ...]:
    """Return concrete asyncio synchronisation primitive types."""
    primitive_types: set[type] = set()
    for attr in _ASYNCIO_PRIMITIVE_NAMES:
        candidate = getattr(asyncio, attr, None)
        if isinstance(candidate, type):
            primitive_types.add(candidate)
    locks_module = getattr(asyncio, "locks", None)
    if locks_module is not None:
        for attr in _ASYNCIO_PRIMITIVE_NAMES:
            candidate = getattr(locks_module, attr, None)
            if isinstance(candidate, type):
                primitive_types.add(candidate)
    return tuple(primitive_types)


_ASYNCIO_PRIMITIVE_TYPES = _collect_asyncio_primitive_types()


SUCCESSFUL_SYMBOLS: set[str] = set()
FAILED_SYMBOLS: set[str] = set()
PEAK_SIMULTANEOUS_WORKERS: int = 0
LAST_RUN_PEAK_SIMULTANEOUS_WORKERS: int = 0
_HOST_PERMITS_HELD: int = 0


if _http_host_limit is not None:
    try:
        _stored_peak = int(_http_host_limit.current_peak())
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        _stored_peak = 0
    else:
        PEAK_SIMULTANEOUS_WORKERS = max(PEAK_SIMULTANEOUS_WORKERS, _stored_peak)


def reset_peak_simultaneous_workers() -> None:
    """Reset ``PEAK_SIMULTANEOUS_WORKERS`` to ``0`` for test isolation."""
    global PEAK_SIMULTANEOUS_WORKERS, LAST_RUN_PEAK_SIMULTANEOUS_WORKERS, _HOST_PERMITS_HELD
    PEAK_SIMULTANEOUS_WORKERS = 0
    LAST_RUN_PEAK_SIMULTANEOUS_WORKERS = 0
    _HOST_PERMITS_HELD = 0


def reset_tracking_state(*, reset_peak: bool = True) -> None:
    """Clear success and failure tracking sets and optionally reset the peak counter."""
    SUCCESSFUL_SYMBOLS.clear()
    FAILED_SYMBOLS.clear()
    global _HOST_PERMITS_HELD
    _HOST_PERMITS_HELD = 0
    if reset_peak:
        reset_peak_simultaneous_workers()


__all__ = [
    "FAILED_SYMBOLS",
    "LAST_RUN_PEAK_SIMULTANEOUS_WORKERS",
    "PEAK_SIMULTANEOUS_WORKERS",
    "SUCCESSFUL_SYMBOLS",
    "reset_peak_simultaneous_workers",
    "reset_tracking_state",
]
