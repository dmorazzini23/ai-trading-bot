from __future__ import annotations

import os  # noqa: F401  # AI-AGENT-REF: env overrides
import time

from .base import (
    get_free_port,
    get_pid_on_port,
    health_check,
    is_market_open,
    market_open_between,
)
from .base import (
    log_warning as _log_warning,
)
from .base import (
    model_lock as _model_lock,
)
from .base import (
    safe_to_datetime as _safe_to_datetime,
)
from .base import (
    validate_ohlcv as _validate_ohlcv,
)
from .timing import HTTP_TIMEOUT, SUBPROCESS_TIMEOUT_S
from .timing import clamp_timeout as _clamp_timeout_new

# Back-compat alias
HTTP_TIMEOUT_DEFAULT = HTTP_TIMEOUT
SUBPROCESS_TIMEOUT_DEFAULT = SUBPROCESS_TIMEOUT_S

# AI-AGENT-REF: relative import to satisfy import contract
try:  # pragma: no cover
    from . import process_manager  # type: ignore
except Exception:  # pragma: no cover
    process_manager = None  # type: ignore


def safe_subprocess_run(
    cmd: list[str] | str,
    *,
    timeout: float | int | None = None,
    **kwargs,
) -> str:
    """Run subprocess and return decoded stdout with clamped timeout."""
    import subprocess  # AI-AGENT-REF: lazy import to respect contract

    to = _clamp_timeout_new(timeout, default_non_test=SUBPROCESS_TIMEOUT_S, min_s=0.1)
    res = subprocess.run(cmd, timeout=to, capture_output=True, **kwargs)
    out = res.stdout
    if isinstance(out, bytes):
        return out.decode(errors="ignore")
    return out or ""


def log_warning(*args, **kwargs):
    return _log_warning(*args, **kwargs)


class _ModelLockProxy:
    _lock = None

    def _ensure(self):
        if self._lock is None:
            self._lock = _model_lock
        return self._lock

    def __enter__(self):
        return self._ensure().__enter__()

    def __exit__(self, *args):
        return self._ensure().__exit__(*args)


model_lock = _ModelLockProxy()


def safe_to_datetime(*args, **kwargs):
    return _safe_to_datetime(*args, **kwargs)


def validate_ohlcv(*args, **kwargs):
    return _validate_ohlcv(*args, **kwargs)


#
# Back-compat wrapper: accept BOTH legacy clamp_timeout(min=, max=, default=)
# and the new clamp_timeout(min_s=, max_s=, default_non_test=, default_test=)
#
def clamp_timeout(
    value: float | int | None = None,
    *,
    # legacy names
    min: float | int | None = None,  # noqa: A002 - back-compat
    max: float | int | None = None,  # noqa: A002 - back-compat
    default: float | int | None = None,
    # new names
    min_s: float | None = None,
    max_s: float | None = None,
    default_non_test: float | None = None,
    default_test: float = 0.25,
):
    # Map legacy -> new if caller used old kw names
    if min_s is None and min is not None:
        min_s = float(min)
    if max_s is None and max is not None:
        max_s = float(max)
    if default_non_test is None and default is not None:
        default_non_test = float(default)
    return _clamp_timeout_new(
        value,
        min_s=min_s if min_s is not None else 0.05,
        max_s=max_s if max_s is not None else 15.0,
        default_non_test=default_non_test if default_non_test is not None else 0.75,
        default_test=default_test,
    )


def psleep(seconds: float) -> None:
    """Plain sleep helper used by tests."""  # AI-AGENT-REF
    time.sleep(seconds)


def sleep_s(seconds: float) -> None:
    """Thin wrapper so tests can monkeypatch easily."""  # AI-AGENT-REF
    time.sleep(_clamp_timeout_new(seconds, default_non_test=0.01, min_s=0.0))


def sleep(seconds: float) -> None:
    """Backward compatible sleep wrapper."""  # AI-AGENT-REF
    sleep_s(seconds)


__all__ = [
    "HTTP_TIMEOUT",
    "HTTP_TIMEOUT_DEFAULT",
    "SUBPROCESS_TIMEOUT_S",
    "SUBPROCESS_TIMEOUT_DEFAULT",
    "clamp_timeout",
    "safe_subprocess_run",
    "log_warning",
    "model_lock",
    "safe_to_datetime",
    "validate_ohlcv",
    "get_free_port",
    "get_pid_on_port",
    "health_check",
    "is_market_open",
    "market_open_between",
    "psleep",
    "sleep_s",
    "sleep",
    "process_manager",
]
