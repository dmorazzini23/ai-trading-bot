"""Process-level locking utilities with xdist-safe file locks."""
from __future__ import annotations
import errno
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from ai_trading.utils.time import monotonic_time
from .timing import sleep as psleep
_LOCKS: dict[str, int] = {}

def _lock_path(name: str) -> Path:
    worker = os.getenv('PYTEST_XDIST_WORKER')
    suffix = f'.{worker}' if worker else ''
    d = Path('/tmp/ai-trading-locks')
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{name}{suffix}.lock'

def acquire_lock(name: str, timeout: float=2.0) -> bool:
    """Non-blocking lock with timeout. Returns True if acquired, False on timeout."""
    path = _lock_path(name)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 384)
    start = monotonic_time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _LOCKS[name] = fd
            return True
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EACCES):
                os.close(fd)
                raise
            if monotonic_time() - start >= timeout:
                os.close(fd)
                return False
            psleep(0.05)

def release_lock(name: str) -> None:
    fd = _LOCKS.pop(name, None)
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

@contextmanager
def file_lock(name: str, timeout: float=2.0):
    ok = acquire_lock(name, timeout=timeout)
    try:
        if not ok:
            raise TimeoutError(f'Could not acquire process lock {name!r} within {timeout}s')
        yield
    finally:
        if ok:
            release_lock(name)

def start_process(name: str) -> dict[str, str]:
    return {'status': 'started', 'name': name}

def stop_process(name: str) -> dict[str, str]:
    return {'status': 'stopped', 'name': name}


__all__ = ['acquire_lock', 'release_lock', 'file_lock', 'start_process', 'stop_process']