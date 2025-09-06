from __future__ import annotations

import asyncio
from typing import Final

from ai_trading.config import management as config

_DEFAULT_LIMIT: Final[int] = 8


def _resolve_limit() -> int:
    try:
        limit = int(config.get_env("AI_TRADING_HOST_LIMIT", str(_DEFAULT_LIMIT), cast=int))
        return max(1, limit)
    except Exception:
        return _DEFAULT_LIMIT


HOST_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(_resolve_limit())


def get_host_semaphore() -> asyncio.Semaphore:
    """Return the global semaphore limiting concurrent host requests."""
    return HOST_SEMAPHORE
