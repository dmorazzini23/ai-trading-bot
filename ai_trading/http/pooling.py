from __future__ import annotations

import asyncio
import os
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

from ai_trading.config import management as config

_DEFAULT_LIMIT: Final[int] = 8

@dataclass(slots=True)
class _SemaphoreRecord:
    semaphore: asyncio.Semaphore
    limit: int
    version: int


@dataclass(slots=True)
class _ResolvedLimitCache:
    raw_env: str | None
    limit: int
    version: int
    config_id: int | None


_HostSemaphoreMap = dict[str, _SemaphoreRecord]
_HOST_SEMAPHORES: WeakKeyDictionary[asyncio.AbstractEventLoop, _HostSemaphoreMap] = (
    WeakKeyDictionary()
)

_LIMIT_CACHE: _ResolvedLimitCache | None = None
_LIMIT_VERSION: int = 0

_DEFAULT_HOST_KEY: Final[str] = "__default__"


def _normalize_host(hostname: str | None) -> str:
    host = (hostname or "").strip().lower()
    return host or _DEFAULT_HOST_KEY


def reset_host_semaphores() -> None:
    """Clear cached host semaphores.

    Useful for ensuring a clean slate when the module is reloaded in tests.
    """

    _HOST_SEMAPHORES.clear()
    invalidate_host_limit_cache()


def _compute_limit(raw: str | None = None) -> int:
    if raw is None:
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


def _resolve_limit() -> tuple[int, int]:
    """Return the current host limit and cache version."""

    global _LIMIT_CACHE, _LIMIT_VERSION

    raw_env = os.getenv("AI_TRADING_HOST_LIMIT")
    config_id: int | None = None

    if raw_env not in (None, ""):
        limit = _compute_limit(raw_env)
    else:
        try:
            cfg = config.get_trading_config()
        except Exception:
            cfg = None
        if cfg is not None:
            config_id = id(cfg)
            try:
                limit_value = getattr(cfg, "host_concurrency_limit", _DEFAULT_LIMIT)
            except Exception:
                limit_value = _DEFAULT_LIMIT
        else:
            limit_value = _DEFAULT_LIMIT
        try:
            limit = max(1, int(limit_value))
        except (TypeError, ValueError):
            limit = _DEFAULT_LIMIT

    cache = _LIMIT_CACHE
    if (
        cache is not None
        and cache.raw_env == raw_env
        and cache.config_id == config_id
        and cache.limit == limit
    ):
        return cache.limit, cache.version

    _LIMIT_VERSION += 1
    version = _LIMIT_VERSION
    _LIMIT_CACHE = _ResolvedLimitCache(
        raw_env=raw_env,
        limit=limit,
        version=version,
        config_id=config_id,
    )
    return limit, version


def get_host_limit() -> int:
    """Return the configured maximum concurrency per host."""

    limit, _ = _resolve_limit()
    return limit


def invalidate_host_limit_cache() -> None:
    """Invalidate cached host limit values.

    The next access will recompute the limit and refresh semaphore records as needed.
    """

    global _LIMIT_CACHE
    _LIMIT_CACHE = None


def _ensure_limit_cache() -> _ResolvedLimitCache:
    """Return a coherent snapshot of the resolved limit cache."""

    global _LIMIT_CACHE

    limit, version = _resolve_limit()
    cache = _LIMIT_CACHE
    if cache is not None and cache.limit == limit and cache.version == version:
        return cache

    raw_env = os.getenv("AI_TRADING_HOST_LIMIT")
    config_id: int | None = None
    if raw_env in (None, ""):
        try:
            cfg = config.get_trading_config()
        except Exception:
            cfg = None
        if cfg is not None:
            config_id = id(cfg)

    # `_resolve_limit` should keep `_LIMIT_CACHE` aligned, but callers may
    # mutate the globals. Rebuild defensively so the snapshot matches the
    # semaphore we create.
    cache = _ResolvedLimitCache(
        raw_env=raw_env,
        limit=limit,
        version=version,
        config_id=config_id,
    )
    _LIMIT_CACHE = cache
    return cache


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
    cache = _ensure_limit_cache()
    resolved_limit = cache.limit
    version = cache.version
    host_map = _get_host_map(loop)
    record = host_map.get(hostname)
    if record is not None:
        if record.limit == resolved_limit and record.version == version:
            return record.semaphore

    semaphore = asyncio.Semaphore(resolved_limit)
    host_map[hostname] = _SemaphoreRecord(semaphore, resolved_limit, version)
    return semaphore


def get_host_semaphore(hostname: str | None = None) -> asyncio.Semaphore:
    """Return the semaphore limiting concurrent host requests for the current loop."""

    loop = asyncio.get_running_loop()
    host = _normalize_host(hostname)
    return _get_or_create_loop_semaphore(loop, host)


def refresh_host_semaphore(hostname: str | None = None) -> asyncio.Semaphore:
    """Force the cached semaphore for the current loop to refresh using the latest limit."""

    loop = asyncio.get_running_loop()
    cache = _ensure_limit_cache()
    limit = cache.limit
    version = cache.version
    host = _normalize_host(hostname)
    semaphore = asyncio.Semaphore(limit)
    host_map = _get_host_map(loop)
    host_map[host] = _SemaphoreRecord(semaphore, limit, version)
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
    "invalidate_host_limit_cache",
    "limit_host",
    "limit_url",
    "refresh_host_semaphore",
    "reset_host_semaphores",
]
