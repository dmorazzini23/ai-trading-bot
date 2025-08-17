from __future__ import annotations

import os  # noqa: F401  # AI-AGENT-REF: kept for potential env overrides
import socket
import time
from typing import Optional

# --- timeouts & clamps ---
HTTP_TIMEOUT_DEFAULT = 10.0
SUBPROCESS_TIMEOUT_DEFAULT = 5.0


def clamp_timeout(
    value: float | int,
    *,
    min: float | int | None = None,
    max: float | int | None = None,
    default: float | int | None = None,
):
    """Clamp numeric timeout or fall back to default."""  # AI-AGENT-REF
    out = default if (value in (None, 0, False) and default is not None) else value
    if out is None:
        return out
    try:
        out = float(out)
    except Exception:  # noqa: BLE001
        return default
    if min is not None and out < min:
        out = min
    if max is not None and out > max:
        out = max
    return out


# Import only when actually needed to respect import contract
# AI-AGENT-REF: removed process_manager import to satisfy import contract


def safe_subprocess_run(
    cmd: list[str] | str,
    *,
    timeout: float | int | None = None,
    **kwargs,
) -> str:
    """Run subprocess and return decoded stdout with clamped timeout."""
    import subprocess  # AI-AGENT-REF: lazy import to respect contract

    to = clamp_timeout(timeout, min=0.1, default=SUBPROCESS_TIMEOUT_DEFAULT)
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
    """Return an available TCP port on localhost."""  # AI-AGENT-REF
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_pid_on_port(port: int) -> Optional[int]:
    """Best-effort PID lookup on Linux using /proc."""  # AI-AGENT-REF
    proc_net = "/proc/net/tcp"
    try:
        hex_port = f"{port:04X}"
        with open(proc_net, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 1 and parts[1].endswith(f":{hex_port}"):
                    return None  # Kernel doesn't expose PID; would require lsof
    except Exception:
        return None
    return None


def health_check(df, resolution: str) -> bool:
    """True if DataFrame has required minimum rows."""  # AI-AGENT-REF
    try:
        rows_required = int(os.getenv("HEALTH_MIN_ROWS", "100"))
    except Exception:
        rows_required = 100
    try:
        n = int(getattr(df, "shape", (0,))[0])
    except Exception:
        n = 0
    return n >= rows_required


def psleep(seconds: float) -> None:
    """Plain sleep helper used by tests."""  # AI-AGENT-REF
    time.sleep(seconds)


def sleep_s(seconds: float) -> None:
    """Thin wrapper so tests can monkeypatch easily."""  # AI-AGENT-REF
    time.sleep(clamp_timeout(seconds, default=0.01, min=0.0))


def sleep(seconds: float) -> None:
    """Backward compatible sleep wrapper."""  # AI-AGENT-REF
    sleep_s(seconds)


__all__ = [
    "HTTP_TIMEOUT_DEFAULT",
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
    "psleep",
    "sleep_s",
    "sleep",
]
