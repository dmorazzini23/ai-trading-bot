from __future__ import annotations

import os  # noqa: F401  # AI-AGENT-REF: kept for potential env overrides
from typing import Any, Callable

# --- timeouts & clamps ---
HTTP_TIMEOUT_DEFAULT = 10.0
SUBPROCESS_TIMEOUT_DEFAULT = 5.0


def clamp_timeout(
    value: float | int | None,
    *,
    default: float,
    min_: float = 0.5,
    max_: float = 60.0,
) -> float:
    if value is None:
        return default
    v = float(value)
    return max(min_, min(max_, v))


# --- lazy process manager accessors ---
def _lazy_process_manager():
    """Import ai_trading.utils.process_manager lazily to satisfy the import contract."""
    import importlib

    mod = importlib.import_module("ai_trading.utils.process_manager")
    return mod


def __getattr__(name: str) -> Any:
    if name == "process_manager":
        mod = _lazy_process_manager()
        globals()["process_manager"] = mod  # cache
        return mod
    raise AttributeError(name)


# Lightweight wrappers to preserve current public surface
def acquire_lock(*args, **kwargs):
    return _lazy_process_manager().acquire_lock(*args, **kwargs)


def release_lock(*args, **kwargs):
    return _lazy_process_manager().release_lock(*args, **kwargs)


def file_lock(*args, **kwargs):
    return _lazy_process_manager().file_lock(*args, **kwargs)


# Import only when actually needed to respect import contract
def get_process_manager():
    from . import process_manager  # local import on demand

    return process_manager


def safe_subprocess_run(
    cmd: list[str] | str,
    *,
    timeout: float | int | None = None,
    **kwargs,
) -> str:
    """Run subprocess and return decoded stdout with clamped timeout."""
    import subprocess  # AI-AGENT-REF: lazy import to respect contract

    to = clamp_timeout(timeout, default=SUBPROCESS_TIMEOUT_DEFAULT)
    res = subprocess.run(cmd, timeout=to, capture_output=True, **kwargs)
    out = res.stdout
    if isinstance(out, bytes):
        return out.decode(errors="ignore")
    return out or ""


__all__ = [
    *sorted(
        set(
            [
                "HTTP_TIMEOUT_DEFAULT",
                "SUBPROCESS_TIMEOUT_DEFAULT",
                "clamp_timeout",
                "get_process_manager",
                "safe_subprocess_run",
                "log_warning",
                "model_lock",
                "safe_to_datetime",
                "validate_ohlcv",
                "process_manager",
                "acquire_lock",
                "release_lock",
                "file_lock",
            ]
        )
    )
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
