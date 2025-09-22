from __future__ import annotations

import asyncio
from typing import Final
from weakref import WeakKeyDictionary

import os

from ai_trading.config import management as config

_DEFAULT_LIMIT: Final[int] = 8

_SemaphoreRecord = tuple[asyncio.Semaphore, int]
_HOST_SEMAPHORES: WeakKeyDictionary[
    asyncio.AbstractEventLoop, _SemaphoreRecord
] = WeakKeyDictionary()


def reset_host_semaphores() -> None:
    """Clear cached host semaphores.

    Useful for ensuring a clean slate when the module is reloaded in tests.
    """

    _HOST_SEMAPHORES.clear()


def _resolve_limit() -> int:
    raw = os.getenv("AI_TRADING_HOST_LIMIT")
    if raw not in (None, ""):
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            pass
    try:
        limit = int(
            config.get_env(
                "AI_TRADING_HOST_LIMIT",
                str(_DEFAULT_LIMIT),
                cast=int,
            )
        )
        return max(1, limit)
    except Exception:
        return _DEFAULT_LIMIT


def _get_or_create_loop_semaphore(
    loop: asyncio.AbstractEventLoop,
) -> asyncio.Semaphore:
    resolved_limit = _resolve_limit()
    record = _HOST_SEMAPHORES.get(loop)
    if record is not None:
        semaphore, cached_limit = record
        if cached_limit == resolved_limit:
            return semaphore

    semaphore = asyncio.Semaphore(resolved_limit)
    _HOST_SEMAPHORES[loop] = (semaphore, resolved_limit)
    return semaphore


def get_host_semaphore() -> asyncio.Semaphore:
    """Return the semaphore limiting concurrent host requests for the current loop."""

    loop = asyncio.get_running_loop()
    return _get_or_create_loop_semaphore(loop)


def refresh_host_semaphore() -> asyncio.Semaphore:
    """Force the cached semaphore for the current loop to refresh using the latest limit."""

    loop = asyncio.get_running_loop()
    limit = _resolve_limit()
    semaphore = asyncio.Semaphore(limit)
    _HOST_SEMAPHORES[loop] = (semaphore, limit)
    return semaphore
