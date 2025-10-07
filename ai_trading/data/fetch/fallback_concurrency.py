from __future__ import annotations

import contextlib
import os
import threading


def _MAX() -> int:
    try:
        return max(1, int(os.getenv("FALLBACK_MAX_CONCURRENCY", "2")))
    except Exception:
        return 2


_sem = threading.BoundedSemaphore(_MAX())
_lock = threading.Lock()
_current = 0
_peak = 0


def _refresh_if_needed() -> None:
    global _sem
    desired = _MAX()
    value = getattr(_sem, "_value", None)
    initial = getattr(_sem, "_initial_value", None)
    if value is not None and initial is not None and initial != desired:
        _sem = threading.BoundedSemaphore(desired)
    elif initial is None and getattr(_sem, "maxvalue", None) not in (None, desired):
        _sem = threading.BoundedSemaphore(desired)


@contextlib.contextmanager
def fallback_slot():
    global _current, _peak
    _refresh_if_needed()
    _sem.acquire()
    try:
        with _lock:
            _current += 1
            if _current > _peak:
                _peak = _current
        yield
    finally:
        with _lock:
            _current -= 1
        _sem.release()


def current_concurrency() -> int:
    with _lock:
        return _current


def peak_concurrency() -> int:
    with _lock:
        return _peak
