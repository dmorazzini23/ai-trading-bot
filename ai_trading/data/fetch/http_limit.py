from __future__ import annotations

import contextlib
import threading

from ai_trading.config.management import get_env

_HOST_LIMITERS: dict[str, tuple[int, threading.BoundedSemaphore]] = {}
_HOST_LOCK = threading.RLock()


def _current_limit() -> int:
    try:
        return max(1, int(get_env("HTTP_HOST_LIMIT", "8", cast=str)))
    except Exception:
        return 8


@contextlib.contextmanager
def acquire_host_slot(host: str):
    limit = _current_limit()
    with _HOST_LOCK:
        current = _HOST_LIMITERS.get(host)
        if current is None or current[0] != limit:
            _HOST_LIMITERS[host] = (limit, threading.BoundedSemaphore(limit))
        semaphore = _HOST_LIMITERS[host][1]
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()
