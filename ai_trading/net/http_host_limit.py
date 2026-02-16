"""Host-level concurrency limit helpers with peak tracking."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AbstractAsyncContextManager, contextmanager
from pathlib import Path
import threading

from ai_trading.logging import get_logger
from ai_trading.http import pooling

logger = get_logger(__name__)

_DEFAULT_HOST_KEY = getattr(pooling, "_DEFAULT_HOST_KEY", "__default__")
_FALLBACK_LOCK = threading.Lock()
_FALLBACK_SYNC_LIMITERS: dict[str, threading.Semaphore] = {}
_FALLBACK_ASYNC_LIMITERS: dict[str, asyncio.Semaphore] = {}

_PEAK_PATH_ENV = "AI_TRADING_FALLBACK_PEAK_PATH"
_DEFAULT_PEAK_PATH = "/tmp/ai_trading_fallback_peak.json"

_COUNTER_LOCK = threading.Lock()
_IN_FLIGHT = 0
_PEAK = 0


def _peak_path() -> Path:
    raw = os.getenv(_PEAK_PATH_ENV)
    if raw:
        try:
            return Path(raw)
        except Exception:
            logger.debug("PEAK_PATH_ENV_PARSE_FAILED", extra={"raw": raw}, exc_info=True)
    return Path(_DEFAULT_PEAK_PATH)


def _load_peak_from_disk() -> int:
    path = _peak_path()
    try:
        data = json.loads(path.read_text())
    except Exception:
        logger.debug("PEAK_FILE_READ_FAILED", extra={"path": str(path)}, exc_info=True)
        return 0
    value = data.get("peak") if isinstance(data, dict) else None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _persist_peak(value: int) -> None:
    path = _peak_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("PEAK_DIR_CREATE_FAILED", extra={"path": str(path.parent)}, exc_info=True)
    tmp_path = path.with_suffix(".tmp")
    payload = json.dumps({"peak": int(value)})
    try:
        tmp_path.write_text(payload)
        tmp_path.replace(path)
    except Exception:
        logger.debug("PEAK_FILE_WRITE_FAILED", extra={"path": str(path)}, exc_info=True)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("PEAK_TEMP_FILE_CLEANUP_FAILED", extra={"path": str(tmp_path)}, exc_info=True)


with _COUNTER_LOCK:
    _PEAK = _load_peak_from_disk()


def record_peak(value: int) -> None:
    """Record a new peak concurrency value if ``value`` exceeds the stored peak."""

    if value <= 0:
        return
    persist_value: int | None = None
    with _COUNTER_LOCK:
        global _PEAK
        if value > _PEAK:
            _PEAK = value
            persist_value = _PEAK
    if persist_value is not None:
        try:
            _persist_peak(persist_value)
        except Exception:
            logger.debug("PEAK_PERSIST_FAILED", extra={"value": persist_value}, exc_info=True)


def _normalize_host(host: str | None) -> str:
    normalizer = getattr(pooling, "_normalize_host", None)
    if callable(normalizer):
        try:
            return normalizer(host)
        except Exception:
            logger.debug("HOST_NORMALIZER_DELEGATE_FAILED", extra={"host": host}, exc_info=True)
    normalized = (host or "").strip().lower()
    return normalized or _DEFAULT_HOST_KEY


def _resolve_fallback_limit() -> int:
    for key in (
        "AI_TRADING_HOST_LIMIT",
        "AI_TRADING_HTTP_HOST_LIMIT",
        "HTTP_MAX_WORKERS",
        "HTTP_MAX_PER_HOST",
    ):
        raw = os.getenv(key)
        if raw not in (None, ""):
            try:
                return max(1, int(raw))
            except (TypeError, ValueError):
                continue
    return max(1, int(getattr(pooling, "_DEFAULT_LIMIT", 1)))


def _get_sync_semaphore(host: str) -> threading.Semaphore:
    getter = getattr(pooling, "get_host_limiter", None)
    if callable(getter):
        try:
            semaphore = getter(host)
        except Exception:
            logger.debug("SYNC_HOST_LIMITER_FETCH_FAILED", extra={"host": host}, exc_info=True)
            semaphore = None
        else:
            if semaphore is not None:
                return semaphore

    with _FALLBACK_LOCK:
        semaphore = _FALLBACK_SYNC_LIMITERS.get(host)
        if semaphore is None:
            semaphore = threading.Semaphore(_resolve_fallback_limit())
            _FALLBACK_SYNC_LIMITERS[host] = semaphore
        return semaphore


def _get_async_semaphore(host: str) -> asyncio.Semaphore:
    getter = getattr(pooling, "get_host_semaphore", None)
    if callable(getter):
        try:
            semaphore = getter(host)
        except Exception:
            logger.debug("ASYNC_HOST_LIMITER_FETCH_FAILED", extra={"host": host}, exc_info=True)
            semaphore = None
        else:
            if semaphore is not None:
                return semaphore

    with _FALLBACK_LOCK:
        semaphore = _FALLBACK_ASYNC_LIMITERS.get(host)
        if semaphore is None:
            semaphore = asyncio.Semaphore(_resolve_fallback_limit())
            _FALLBACK_ASYNC_LIMITERS[host] = semaphore
        return semaphore


def _register_acquire() -> None:
    global _IN_FLIGHT
    with _COUNTER_LOCK:
        _IN_FLIGHT += 1
        current = _IN_FLIGHT
    record_peak(current)


def _register_release() -> None:
    global _IN_FLIGHT
    with _COUNTER_LOCK:
        _IN_FLIGHT = max(0, _IN_FLIGHT - 1)


class _TrackedAsyncLimiter(AbstractAsyncContextManager["_TrackedAsyncLimiter"]):
    """Async context manager that tracks inflight permit usage."""

    __slots__ = ("_inner", "_acquired", "_tracks_inflight")

    def __init__(self, host: str | None) -> None:
        self._inner, self._tracks_inflight = _build_async_limiter(host)
        self._acquired = False

    async def __aenter__(self) -> "_TrackedAsyncLimiter":
        await self._inner.__aenter__()
        if not self._tracks_inflight:
            _register_acquire()
        self._acquired = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        try:
            if self._acquired and not self._tracks_inflight:
                _register_release()
        finally:
            return await self._inner.__aexit__(exc_type, exc, tb)


class _FallbackAsyncLimiter(AbstractAsyncContextManager["_FallbackAsyncLimiter"]):
    __slots__ = ("_host", "_semaphore", "_acquired")

    def __init__(self, host: str | None) -> None:
        self._host = _normalize_host(host)
        self._semaphore: asyncio.Semaphore | None = None
        self._acquired = False

    async def __aenter__(self) -> "_FallbackAsyncLimiter":
        semaphore = _get_async_semaphore(self._host)
        self._semaphore = semaphore
        await semaphore.acquire()
        _register_acquire()
        self._acquired = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        try:
            if self._semaphore is not None and self._acquired:
                try:
                    self._semaphore.release()
                except ValueError:
                    pass
                _register_release()
        finally:
            return None


def _build_async_limiter(host: str | None) -> tuple[AbstractAsyncContextManager[object], bool]:
    limiter_cls = getattr(pooling, "AsyncHostLimiter", None)
    if callable(limiter_cls):
        try:
            limiter = limiter_cls(host)
            return limiter, False  # type: ignore[return-value]
        except Exception:
            logger.debug("ASYNC_HOST_LIMITER_BUILD_FAILED", extra={"host": host}, exc_info=True)
    return _FallbackAsyncLimiter(host), True


def host_limiter_async(host: str | None = None) -> _TrackedAsyncLimiter:
    """Return an async host limiter that tracks global peak usage."""

    return _TrackedAsyncLimiter(host)


@contextmanager
def host_limiter(host: str | None = None):
    """Synchronously limit host concurrency while tracking peaks."""

    normalized = _normalize_host(host)
    semaphore = _get_sync_semaphore(normalized)
    semaphore.acquire()
    _register_acquire()
    try:
        yield
    finally:
        try:
            semaphore.release()
        except ValueError:
            pass
        finally:
            _register_release()


def current_inflight() -> int:
    """Return the number of currently active host-limited operations."""

    with _COUNTER_LOCK:
        return _IN_FLIGHT


def current_peak() -> int:
    """Return the highest observed concurrent host operations."""

    with _COUNTER_LOCK:
        return _PEAK


__all__ = [
    "current_inflight",
    "current_peak",
    "host_limiter",
    "host_limiter_async",
    "record_peak",
]
