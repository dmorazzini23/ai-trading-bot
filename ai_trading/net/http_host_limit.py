"""Host-level concurrency limit helpers with peak tracking."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AbstractAsyncContextManager, contextmanager
from pathlib import Path
import threading
from ai_trading.http import pooling

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
            pass
    return Path(_DEFAULT_PEAK_PATH)


def _load_peak_from_disk() -> int:
    path = _peak_path()
    try:
        data = json.loads(path.read_text())
    except Exception:
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
        pass
    tmp_path = path.with_suffix(".tmp")
    payload = json.dumps({"peak": int(value)})
    try:
        tmp_path.write_text(payload)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


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
            pass


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

    __slots__ = ("_inner", "_acquired")

    def __init__(self, host: str | None) -> None:
        self._inner = pooling.AsyncHostLimiter(host)
        self._acquired = False

    async def __aenter__(self) -> "_TrackedAsyncLimiter":
        await self._inner.__aenter__()
        _register_acquire()
        self._acquired = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool | None:
        try:
            if self._acquired:
                _register_release()
        finally:
            return await self._inner.__aexit__(exc_type, exc, tb)


def host_limiter_async(host: str | None = None) -> _TrackedAsyncLimiter:
    """Return an async host limiter that tracks global peak usage."""

    return _TrackedAsyncLimiter(host)


@contextmanager
def host_limiter(host: str | None = None):
    """Synchronously limit host concurrency while tracking peaks."""

    normalized = pooling._normalize_host(host)  # type: ignore[attr-defined]
    semaphore = pooling.get_host_limiter(normalized)
    semaphore.acquire()
    _register_acquire()
    try:
        yield
    finally:
        try:
            semaphore.release()
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

