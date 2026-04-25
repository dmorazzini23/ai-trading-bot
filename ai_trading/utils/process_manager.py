"""Process-level locking utilities with xdist-safe file locks."""
from __future__ import annotations
import errno
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from ai_trading.config.management import get_env
from ai_trading.utils.time import monotonic_time
from .timing import sleep as psleep
_LOCKS: dict[str, int] = {}

def _lock_path(name: str, *, force_user_dir: bool = False) -> Path:
    worker = get_env("PYTEST_XDIST_WORKER", "", cast=str, resolve_aliases=False)
    suffix = f'.{worker}' if worker else ''
    primary = Path('/tmp/ai-trading-locks')
    user_dir = Path(tempfile.gettempdir()) / f'ai-trading-locks-{os.getuid()}'
    d = user_dir if force_user_dir else primary
    try:
        d.mkdir(parents=True, exist_ok=True)
        if not os.access(d, os.W_OK | os.X_OK):
            d = user_dir
            d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = user_dir
        d.mkdir(parents=True, exist_ok=True)
    return d / f'{name}{suffix}.lock'

def acquire_lock(name: str, timeout: float=2.0) -> bool:
    """Non-blocking lock with timeout. Returns True if acquired, False on timeout."""
    path = _lock_path(name)
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 384)
    except PermissionError:
        path = _lock_path(name, force_user_dir=True)
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
