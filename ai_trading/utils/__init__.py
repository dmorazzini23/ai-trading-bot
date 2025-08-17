from __future__ import annotations

import os

# --- timeouts & clamps ---
HTTP_DEFAULT_TIMEOUT: float = float(os.environ.get("AI_HTTP_TIMEOUT", "10"))  # seconds
SUBPROCESS_DEFAULT_TIMEOUT: float = float(os.environ.get("AI_SUBPROC_TIMEOUT", "3"))


def clamp_timeout(value: float | int, min_s: float = 0.5, max_s: float = 60.0) -> float:
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return HTTP_DEFAULT_TIMEOUT
    return max(min_s, min(v, max_s))


# Import only when actually needed to respect import contract
def get_process_manager():
    from . import process_manager  # local import on demand

    return process_manager


__all__ = [
    "HTTP_DEFAULT_TIMEOUT",
    "SUBPROCESS_DEFAULT_TIMEOUT",
    "clamp_timeout",
    "get_process_manager",
    "log_warning",
    "model_lock",
    "safe_to_datetime",
    "validate_ohlcv",
]


def log_warning(*args, **kwargs):
    from .base import log_warning as _log_warning

    return _log_warning(*args, **kwargs)


class _ModelLockProxy:
    _lock = None

    def _ensure(self):
        if self._lock is None:
            from .base import model_lock as _model_lock

            self._lock = _model_lock
        return self._lock

    def __enter__(self):
        return self._ensure().__enter__()

    def __exit__(self, *args):
        return self._ensure().__exit__(*args)


model_lock = _ModelLockProxy()


def safe_to_datetime(*args, **kwargs):
    from .base import safe_to_datetime as _safe_to_datetime

    return _safe_to_datetime(*args, **kwargs)


def validate_ohlcv(*args, **kwargs):
    from .base import validate_ohlcv as _validate_ohlcv

    return _validate_ohlcv(*args, **kwargs)
