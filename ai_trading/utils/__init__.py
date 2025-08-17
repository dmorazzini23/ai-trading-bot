from __future__ import annotations

import os  # noqa: F401  # AI-AGENT-REF: kept for potential env overrides
import socket
from contextlib import closing

import pandas as pd

try:  # pragma: no cover - optional runtime dependency
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore[assignment]

# --- timeouts & clamps ---
HTTP_TIMEOUT_DEFAULT = 10.0
SUBPROCESS_TIMEOUT_DEFAULT = 5.0


def clamp_timeout(
    value: float | int | None,
    default: float | int | None = None,
    *,
    min_: float = 0.5,
    max_: float = 60.0,
) -> float:
    """Return a sane timeout clamped between bounds."""
    if value is None:
        value = default
    try:
        v = float(value) if value is not None else 10.0
    except Exception:  # noqa: BLE001
        v = float(default) if default is not None else 10.0
    return max(min_, min(max_, v))


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


def get_free_port() -> int:
    """Return an ephemeral free TCP port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def get_pid_on_port(port: int) -> int | None:
    """Return PID bound to ``port`` if available."""
    if psutil is None:
        return None
    for conn in psutil.net_connections(kind="inet"):
        try:
            if conn.laddr and conn.laddr.port == port and conn.pid:
                return int(conn.pid)
        except Exception:  # noqa: BLE001
            continue
    return None


def health_check(df: pd.DataFrame, resolution: str) -> bool:
    """Minimal dataframe health check used in tests."""
    try:
        min_rows = int(os.getenv("HEALTH_MIN_ROWS", "100"))
    except Exception:  # noqa: BLE001
        min_rows = 100
    return int(getattr(df, "shape", (0,))[0]) >= min_rows


def sleep(seconds: float) -> None:
    from .time import sleep as _sleep

    _sleep(seconds)


__all__ = [
    "HTTP_TIMEOUT_DEFAULT",
    "SUBPROCESS_TIMEOUT_DEFAULT",
    "clamp_timeout",
    "get_process_manager",
    "safe_subprocess_run",
    "log_warning",
    "model_lock",
    "safe_to_datetime",
    "validate_ohlcv",
    "get_free_port",
    "get_pid_on_port",
    "health_check",
    "sleep",
]
