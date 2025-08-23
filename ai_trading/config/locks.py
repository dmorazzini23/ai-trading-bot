"""Threading utilities for configuration management."""
import threading

class LockWithTimeout:
    """Simple wrapper around ``threading.Lock`` with timeout support."""

    def __init__(self, timeout: float=5.0):
        self._lock = threading.Lock()
        self._timeout = timeout

    def acquire(self) -> bool:
        return self._lock.acquire(timeout=self._timeout)

    def release(self) -> None:
        if self._lock.locked():
            self._lock.release()

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError('Lock acquisition timed out')
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
__all__ = ['LockWithTimeout']