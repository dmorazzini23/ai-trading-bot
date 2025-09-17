from __future__ import annotations

import asyncio
from typing import Final
from weakref import WeakKeyDictionary

from ai_trading.config import management as config

_DEFAULT_LIMIT: Final[int] = 8

_SemaphoreRecord = tuple[asyncio.Semaphore, int]
_HOST_SEMAPHORES: WeakKeyDictionary[
    asyncio.AbstractEventLoop, _SemaphoreRecord
] = WeakKeyDictionary()


def _resolve_limit() -> int:
    try:
        limit = int(config.get_env("AI_TRADING_HOST_LIMIT", str(_DEFAULT_LIMIT), cast=int))
        return max(1, limit)
    except Exception:
        return _DEFAULT_LIMIT


def _get_or_create_loop_semaphore(
    loop: asyncio.AbstractEventLoop, limit: int
) -> asyncio.Semaphore:
    record = _HOST_SEMAPHORES.get(loop)
    if record is not None:
        semaphore, cached_limit = record
        if cached_limit == limit:
            return semaphore

    semaphore = asyncio.Semaphore(limit)
    _HOST_SEMAPHORES[loop] = (semaphore, limit)
    return semaphore


def get_host_semaphore() -> asyncio.Semaphore:
    """Return the semaphore limiting concurrent host requests for the current loop."""

    loop = asyncio.get_running_loop()
    limit = _resolve_limit()
    return _get_or_create_loop_semaphore(loop, limit)


def refresh_host_semaphore() -> asyncio.Semaphore:
    """Force the cached semaphore for the current loop to refresh using the latest limit."""

    loop = asyncio.get_running_loop()
    limit = _resolve_limit()
    record = _HOST_SEMAPHORES.get(loop)
    if record is not None and record[1] == limit:
        return record[0]

    semaphore = asyncio.Semaphore(limit)
    _HOST_SEMAPHORES[loop] = (semaphore, limit)
    return semaphore
