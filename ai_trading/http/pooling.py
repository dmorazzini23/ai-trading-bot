from __future__ import annotations

import asyncio
import os
from contextlib import AbstractAsyncContextManager
from typing import Final
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

from ai_trading.config import management as config

_DEFAULT_LIMIT: Final[int] = 8

_SemaphoreRecord = tuple[asyncio.Semaphore, int]
_HostSemaphoreMap = dict[str, _SemaphoreRecord]
_HOST_SEMAPHORES: WeakKeyDictionary[asyncio.AbstractEventLoop, _HostSemaphoreMap] = (
    WeakKeyDictionary()
)

_DEFAULT_HOST_KEY: Final[str] = "__default__"


def _normalize_host(hostname: str | None) -> str:
    host = (hostname or "").strip().lower()
    return host or _DEFAULT_HOST_KEY


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


def get_host_limit() -> int:
    """Return the configured maximum concurrency per host."""

    return _resolve_limit()


def _get_host_map(loop: asyncio.AbstractEventLoop) -> _HostSemaphoreMap:
    host_map = _HOST_SEMAPHORES.get(loop)
    if host_map is None:
        host_map = {}
        _HOST_SEMAPHORES[loop] = host_map
    return host_map


def _get_or_create_loop_semaphore(
    loop: asyncio.AbstractEventLoop,
    hostname: str,
) -> asyncio.Semaphore:
    resolved_limit = _resolve_limit()
    host_map = _get_host_map(loop)
    record = host_map.get(hostname)
    if record is not None:
        semaphore, cached_limit = record
        if cached_limit == resolved_limit:
            return semaphore

    semaphore = asyncio.Semaphore(resolved_limit)
    host_map[hostname] = (semaphore, resolved_limit)
    return semaphore


def get_host_semaphore(hostname: str | None = None) -> asyncio.Semaphore:
    """Return the semaphore limiting concurrent host requests for the current loop."""

    loop = asyncio.get_running_loop()
    host = _normalize_host(hostname)
    return _get_or_create_loop_semaphore(loop, host)


def refresh_host_semaphore(hostname: str | None = None) -> asyncio.Semaphore:
    """Force the cached semaphore for the current loop to refresh using the latest limit."""

    loop = asyncio.get_running_loop()
    limit = _resolve_limit()
    host = _normalize_host(hostname)
    semaphore = asyncio.Semaphore(limit)
    host_map = _get_host_map(loop)
    host_map[host] = (semaphore, limit)
    return semaphore


class AsyncHostLimiter(AbstractAsyncContextManager["AsyncHostLimiter"]):
    """Async context manager that bounds concurrent requests per hostname."""

    __slots__ = ("_host", "_semaphore")

    def __init__(self, host: str | None) -> None:
        self._host = _normalize_host(host)
        self._semaphore: asyncio.Semaphore | None = None

    @classmethod
    def from_url(cls, url: str) -> "AsyncHostLimiter":
        parsed = urlparse(url)
        host = parsed.hostname or parsed.netloc or None
        return cls(host)

    async def __aenter__(self) -> "AsyncHostLimiter":
        semaphore = get_host_semaphore(self._host)
        self._semaphore = semaphore
        await semaphore.acquire()
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._semaphore is not None:
            self._semaphore.release()
            self._semaphore = None


def limit_host(hostname: str | None = None) -> AsyncHostLimiter:
    """Return an :class:`AsyncHostLimiter` for ``hostname``."""

    return AsyncHostLimiter(hostname)


def limit_url(url: str) -> AsyncHostLimiter:
    """Return an :class:`AsyncHostLimiter` keyed by ``url``'s hostname."""

    return AsyncHostLimiter.from_url(url)


__all__ = [
    "AsyncHostLimiter",
    "get_host_limit",
    "get_host_semaphore",
    "limit_host",
    "limit_url",
    "refresh_host_semaphore",
    "reset_host_semaphores",
]
