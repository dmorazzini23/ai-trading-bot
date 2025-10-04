from __future__ import annotations

import asyncio
import importlib
import os
import sys
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from types import ModuleType
from typing import Final, NamedTuple
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

_previous_host_semaphores = globals().get("_HOST_SEMAPHORES")
if isinstance(_previous_host_semaphores, WeakKeyDictionary):
    _previous_host_semaphores.clear()
elif hasattr(_previous_host_semaphores, "clear"):
    try:
        _previous_host_semaphores.clear()  # type: ignore[call-arg]  # pragma: no cover - defensive guard for exotic reload state
    except Exception:  # pragma: no cover - defensive guard for exotic reload state
        pass

if "_LIMIT_CACHE" in globals():
    globals()["_LIMIT_CACHE"] = None

if "_LIMIT_VERSION" in globals():
    globals()["_LIMIT_VERSION"] = 0

from ai_trading.config import management as config

_DEFAULT_LIMIT: Final[int] = 8

@dataclass(slots=True)
class _SemaphoreRecord:
    semaphore: asyncio.Semaphore
    limit: int
    version: int


@dataclass(slots=True)
class _ResolvedLimitCache:
    env_key: str | None
    raw_env: str | None
    limit: int
    version: int
    config_id: int | None
    env_snapshot: tuple[str | None, str | None, str | None]


class HostLimitSnapshot(NamedTuple):
    """Lightweight snapshot describing the current host limit state."""

    limit: int
    version: int


_HostSemaphoreMap = dict[str, _SemaphoreRecord]
_HOST_SEMAPHORES: WeakKeyDictionary[asyncio.AbstractEventLoop, _HostSemaphoreMap] = (
    WeakKeyDictionary()
)
_RETIRED_SEMAPHORES: list[asyncio.Semaphore] = []

_LIMIT_CACHE: _ResolvedLimitCache | None = None
_LIMIT_VERSION: int = 0
_LAST_LIMIT_ENV_SNAPSHOT: tuple[str | None, str | None, str | None] | None = None

_ENV_LIMIT_KEYS: Final[tuple[str, str, str]] = (
    "HTTP_MAX_PER_HOST",
    "AI_TRADING_HTTP_HOST_LIMIT",
    "AI_TRADING_HOST_LIMIT",
)

_DEFAULT_HOST_KEY: Final[str] = "__default__"


def _load_fallback_concurrency_module() -> ModuleType | None:
    module = sys.modules.get("ai_trading.data.fallback.concurrency")
    if module is not None:
        return module
    try:
        return importlib.import_module("ai_trading.data.fallback.concurrency")
    except Exception:
        return None


def _normalise_pooling_state(state: object | None) -> tuple[int, int] | None:
    if state is None:
        return None
    if isinstance(state, tuple) and len(state) >= 2:
        limit, version = state[0], state[1]
    else:
        limit = getattr(state, "limit", None)
        version = getattr(state, "version", None)
    try:
        limit = int(limit)
        version = int(version)
    except (TypeError, ValueError):
        return None
    if limit < 1:
        limit = 1
    return limit, version


def _get_pooling_limit_state() -> tuple[int, int] | None:
    module = _load_fallback_concurrency_module()
    if module is None:
        return None
    state = getattr(module, "_POOLING_LIMIT_STATE", None)
    return _normalise_pooling_state(state)


def _set_pooling_limit_state(limit: int, version: int) -> None:
    module = _load_fallback_concurrency_module()
    if module is None:
        return
    recorder = getattr(module, "_record_pooling_snapshot", None)
    if callable(recorder):
        try:
            recorder(limit, version)
            return
        except Exception:
            pass
    try:
        module._POOLING_LIMIT_STATE = (max(1, int(limit)), int(version))  # type: ignore[attr-defined]
    except Exception:
        return
    local_version = getattr(module, "_LOCAL_POOLING_VERSION", None)
    if isinstance(local_version, int) and version > local_version:
        try:
            module._LOCAL_POOLING_VERSION = version  # type: ignore[attr-defined]
        except Exception:
            pass


def _sync_limit_cache_from_pooling(limit: int, version: int) -> HostLimitSnapshot:
    global _LIMIT_CACHE, _LIMIT_VERSION

    env_snapshot = tuple(os.getenv(key) for key in _ENV_LIMIT_KEYS)
    cache = _LIMIT_CACHE
    limit = max(1, int(limit))
    version = int(version)
    if cache is None:
        cache = _ResolvedLimitCache(
            env_key=None,
            raw_env=None,
            limit=limit,
            version=version,
            config_id=None,
            env_snapshot=env_snapshot,
        )
    else:
        cache = _ResolvedLimitCache(
            env_key=cache.env_key,
            raw_env=cache.raw_env,
            limit=limit,
            version=version,
            config_id=cache.config_id,
            env_snapshot=env_snapshot,
        )
    _LIMIT_CACHE = cache
    if version > _LIMIT_VERSION or _LIMIT_VERSION <= 0:
        _LIMIT_VERSION = version
    return HostLimitSnapshot(limit, version)


def _normalize_host(hostname: str | None) -> str:
    host = (hostname or "").strip().lower()
    return host or _DEFAULT_HOST_KEY


def _annotate_semaphore_metadata(
    semaphore: asyncio.Semaphore, limit: int, version: int
) -> None:
    try:
        setattr(semaphore, "_ai_trading_host_limit", limit)
        setattr(semaphore, "_ai_trading_host_limit_version", version)
    except Exception:  # pragma: no cover - attribute assignment failure should not break runtime
        pass


def _build_semaphore(limit: int, version: int) -> asyncio.Semaphore:
    semaphore = asyncio.Semaphore(limit)
    _annotate_semaphore_metadata(semaphore, limit, version)
    return semaphore


def _clear_all_loop_semaphores() -> None:
    """Remove cached host semaphores for all tracked event loops."""

    for loop, host_map in list(_HOST_SEMAPHORES.items()):
        try:
            host_map.clear()
        except Exception:
            _HOST_SEMAPHORES.pop(loop, None)


def reset_host_semaphores(*, clear_limit_cache: bool = True) -> None:
    """Clear cached host semaphores and, optionally, the limit cache.

    Resetting the semaphore cache should also advance the cached limit
    version so that subsequent callers pick up freshly created semaphore
    instances even when the resolved limit value itself has not changed.
    """

    global _LIMIT_CACHE, _LIMIT_VERSION

    if _HOST_SEMAPHORES:
        for host_map in list(_HOST_SEMAPHORES.values()):
            for record in host_map.values():
                _RETIRED_SEMAPHORES.append(record.semaphore)
    _RETIRED_SEMAPHORES[:] = _RETIRED_SEMAPHORES[-8:]
    _HOST_SEMAPHORES.clear()

    # Bump the global version so that future semaphore lookups will treat the
    # cleared cache as a new generation. When we retain the limit cache we
    # rewrite it with the updated version to keep the metadata consistent.
    _LIMIT_VERSION += 1

    if clear_limit_cache:
        _LIMIT_CACHE = None
    elif _LIMIT_CACHE is not None:
        cache = _LIMIT_CACHE
        _LIMIT_CACHE = _ResolvedLimitCache(
            env_key=cache.env_key,
            raw_env=cache.raw_env,
            limit=cache.limit,
            version=_LIMIT_VERSION,
            config_id=cache.config_id,
            env_snapshot=cache.env_snapshot,
        )


def testing_reset_host_semaphores() -> None:
    """Test helper: clear cached host semaphores without dropping the limit cache."""

    reset_host_semaphores(clear_limit_cache=False)


def testing_reset_host_limits() -> None:
    """Test helper: clear cached semaphores and force the limit to recompute."""

    reset_host_semaphores(clear_limit_cache=True)


def _compute_limit(raw: str | None = None) -> int:
    if raw is None:
        raw = (
            os.getenv("HTTP_MAX_PER_HOST")
            or os.getenv("AI_TRADING_HTTP_HOST_LIMIT")
            or os.getenv("AI_TRADING_HOST_LIMIT")
        )
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


def _read_limit_source(
    env_snapshot: tuple[str | None, str | None, str | None]
) -> tuple[int, str | None, str | None, int | None]:
    """Return the resolved limit and metadata describing its source."""

    for env_key, raw_env in zip(_ENV_LIMIT_KEYS, env_snapshot):
        if raw_env not in (None, ""):
            limit = _compute_limit(raw_env)
            return limit, env_key, raw_env, None

    try:
        cfg = config.get_trading_config()
    except Exception:
        cfg = None

    config_id: int | None = None
    limit_value = _DEFAULT_LIMIT
    if cfg is not None:
        config_id = id(cfg)
        try:
            limit_value = getattr(cfg, "host_concurrency_limit", _DEFAULT_LIMIT)
        except Exception:
            limit_value = _DEFAULT_LIMIT

    try:
        limit = max(1, int(limit_value))
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT

    return limit, None, None, config_id


def _refresh_all_host_semaphores(snapshot: HostLimitSnapshot) -> None:
    """Rebuild cached semaphores for all known event loops."""

    if not _HOST_SEMAPHORES:
        _set_pooling_limit_state(snapshot.limit, snapshot.version)
        return

    stale_loops: list[asyncio.AbstractEventLoop] = []
    for loop, host_map in list(_HOST_SEMAPHORES.items()):
        if not host_map:
            if getattr(loop, "is_closed", None) and loop.is_closed():
                stale_loops.append(loop)
            continue
        for host in list(host_map.keys()):
            refresh_host_semaphore(
                host,
                loop=loop,
                snapshot=snapshot,
                update_pooling_state=False,
            )
    for loop in stale_loops:
        _HOST_SEMAPHORES.pop(loop, None)
    _set_pooling_limit_state(snapshot.limit, snapshot.version)


def _resolve_limit() -> tuple[int, int]:
    """Return the current host limit and cache version."""

    global _LIMIT_CACHE, _LIMIT_VERSION, _LAST_LIMIT_ENV_SNAPSHOT

    env_snapshot = tuple(os.getenv(key) for key in _ENV_LIMIT_KEYS)
    prior_cache = _LIMIT_CACHE
    env_changed = _LAST_LIMIT_ENV_SNAPSHOT != env_snapshot
    if env_changed:
        _LIMIT_CACHE = None

    limit, env_key, raw_env, config_id = _read_limit_source(env_snapshot)

    cache = _LIMIT_CACHE
    if (
        cache is not None
        and cache.limit == limit
        and cache.env_key == env_key
        and cache.raw_env == raw_env
        and cache.config_id == config_id
        and cache.env_snapshot == env_snapshot
    ):
        _LAST_LIMIT_ENV_SNAPSHOT = env_snapshot
        return cache.limit, cache.version

    if _LIMIT_VERSION == 0:
        _LIMIT_VERSION = 1
    else:
        _LIMIT_VERSION += 1
    version = _LIMIT_VERSION
    _LIMIT_CACHE = _ResolvedLimitCache(
        env_key=env_key,
        raw_env=raw_env,
        limit=limit,
        version=version,
        config_id=config_id,
        env_snapshot=env_snapshot,
    )
    snapshot = HostLimitSnapshot(limit, version)
    _LAST_LIMIT_ENV_SNAPSHOT = env_snapshot

    should_refresh = env_changed
    if not should_refresh and prior_cache is not None:
        should_refresh = (
            prior_cache.limit != limit
            or prior_cache.env_key != env_key
            or prior_cache.raw_env != raw_env
            or prior_cache.config_id != config_id
        )

    if should_refresh:
        _refresh_all_host_semaphores(snapshot)

    return limit, version


def _ensure_limit_cache() -> _ResolvedLimitCache:
    """Return a coherent snapshot of the resolved limit cache."""

    global _LIMIT_CACHE

    _resolve_limit()
    cache = _LIMIT_CACHE
    if cache is None:
        # As a last resort, rebuild a minimal cache so downstream code can
        # proceed. This should be rare and indicates external mutation.
        limit, version = _resolve_limit()
        cache = _ResolvedLimitCache(
            env_key=None,
            raw_env=None,
            limit=limit,
            version=version,
            config_id=None,
            env_snapshot=tuple(os.getenv(key) for key in _ENV_LIMIT_KEYS),
        )
        _LIMIT_CACHE = cache
    return cache


def get_host_limit_snapshot() -> HostLimitSnapshot:
    """Return the current host limit together with its cache version."""

    cache = _ensure_limit_cache()
    return HostLimitSnapshot(cache.limit, cache.version)


def get_host_limit() -> int:
    """Return the configured maximum concurrency per host."""

    snapshot = get_host_limit_snapshot()
    return snapshot.limit


def invalidate_host_limit_cache() -> None:
    """Invalidate cached host limit values.

    The next access will recompute the limit and refresh semaphore records as needed.
    """

    global _LIMIT_CACHE
    _LIMIT_CACHE = None
    limit, version = _resolve_limit()
    snapshot = HostLimitSnapshot(limit, version)
    _refresh_all_host_semaphores(snapshot)


def reload_host_limit_if_env_changed() -> HostLimitSnapshot:
    """Refresh cached limit metadata when relevant environment variables change."""

    global _LAST_LIMIT_ENV_SNAPSHOT, _LIMIT_CACHE

    env_snapshot = tuple(os.getenv(key) for key in _ENV_LIMIT_KEYS)
    cache = _LIMIT_CACHE
    if cache is not None and cache.env_snapshot == env_snapshot:
        _LAST_LIMIT_ENV_SNAPSHOT = env_snapshot
        snapshot = HostLimitSnapshot(cache.limit, cache.version)
        _set_pooling_limit_state(snapshot.limit, snapshot.version)
        return snapshot

    if cache is not None and cache.env_snapshot != env_snapshot:
        _clear_all_loop_semaphores()

    limit, version = _resolve_limit()
    cache = _LIMIT_CACHE
    if cache is not None:
        snapshot = HostLimitSnapshot(cache.limit, cache.version)
    else:
        snapshot = HostLimitSnapshot(limit, version)
    _LAST_LIMIT_ENV_SNAPSHOT = env_snapshot
    _set_pooling_limit_state(snapshot.limit, snapshot.version)
    return snapshot


def _get_host_map(loop: asyncio.AbstractEventLoop) -> _HostSemaphoreMap:
    host_map = _HOST_SEMAPHORES.get(loop)
    if host_map is None:
        host_map = {}
        _HOST_SEMAPHORES[loop] = host_map
    return host_map


def _get_or_create_loop_semaphore(
    loop: asyncio.AbstractEventLoop,
    hostname: str,
    snapshot: HostLimitSnapshot | None = None,
) -> asyncio.Semaphore:
    if snapshot is None:
        cache = _ensure_limit_cache()
        resolved_limit = cache.limit
        version = cache.version
    else:
        resolved_limit = snapshot.limit
        version = snapshot.version
    host_map = _get_host_map(loop)

    if host_map:
        if any(record.version != version for record in host_map.values()):
            stale_hosts = list(host_map.keys())
        else:
            stale_hosts = [
                host
                for host, record in list(host_map.items())
                if record.limit != resolved_limit
            ]
        for host in stale_hosts:
            host_map[host] = _SemaphoreRecord(
                _build_semaphore(resolved_limit, version), resolved_limit, version
            )

    record = host_map.get(hostname)
    if record is not None and record.limit == resolved_limit and record.version == version:
        _annotate_semaphore_metadata(record.semaphore, resolved_limit, version)
        return record.semaphore

    semaphore = _build_semaphore(resolved_limit, version)
    host_map[hostname] = _SemaphoreRecord(semaphore, resolved_limit, version)
    return semaphore


def get_host_semaphore(hostname: str | None = None) -> asyncio.Semaphore:
    """Return the semaphore limiting concurrent host requests for the current loop."""

    loop = asyncio.get_running_loop()
    host = _normalize_host(hostname)
    reload_host_limit_if_env_changed()
    snapshot = get_host_limit_snapshot()
    pooling_state = _get_pooling_limit_state()
    if pooling_state is not None:
        pooling_limit, pooling_version = pooling_state
        if (
            pooling_version > snapshot.version
            or (
                pooling_version == snapshot.version
                and pooling_limit != snapshot.limit
            )
        ):
            snapshot = _sync_limit_cache_from_pooling(
                pooling_limit, pooling_version
            )
        elif pooling_version < snapshot.version:
            _set_pooling_limit_state(snapshot.limit, snapshot.version)
    else:
        _set_pooling_limit_state(snapshot.limit, snapshot.version)
    return _get_or_create_loop_semaphore(loop, host, snapshot)


def refresh_host_semaphore(
    hostname: str | None = None,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    snapshot: HostLimitSnapshot | None = None,
    update_pooling_state: bool = True,
) -> asyncio.Semaphore:
    """Force the cached semaphore for the current loop to refresh using the latest limit."""

    if loop is None:
        loop = asyncio.get_running_loop()
    host = _normalize_host(hostname)
    host_map = _get_host_map(loop)
    if snapshot is None:
        cache = _ensure_limit_cache()
        snapshot = HostLimitSnapshot(cache.limit, cache.version)
    previous = host_map.get(host)
    semaphore = _build_semaphore(snapshot.limit, snapshot.version)
    host_map[host] = _SemaphoreRecord(semaphore, snapshot.limit, snapshot.version)
    if previous is not None:
        _RETIRED_SEMAPHORES.append(previous.semaphore)
        if len(_RETIRED_SEMAPHORES) > 8:
            _RETIRED_SEMAPHORES.pop(0)
    if update_pooling_state:
        _set_pooling_limit_state(snapshot.limit, snapshot.version)
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
    "HostLimitSnapshot",
    "get_host_limit",
    "get_host_limit_snapshot",
    "get_host_semaphore",
    "invalidate_host_limit_cache",
    "limit_host",
    "limit_url",
    "reload_host_limit_if_env_changed",
    "refresh_host_semaphore",
    "reset_host_semaphores",
    "testing_reset_host_limits",
    "testing_reset_host_semaphores",
]
