from __future__ import annotations

import os  # noqa: F401  # AI-AGENT-REF: kept for potential env overrides
import socket
import time

# AI-AGENT-REF: psutil imported lazily to respect import contract

# --- timeouts & clamps ---
HTTP_TIMEOUT_DEFAULT = 10.0
SUBPROCESS_TIMEOUT_DEFAULT = 5.0


def clamp_timeout(
    seconds: float,
    *,
    default: float = 5.0,
    min_s: float = 0.1,
    max_s: float = 60.0,
) -> float:
    """Clamp ``seconds`` within bounds or fall back to ``default``."""  # AI-AGENT-REF
    try:
        sec = float(seconds)
        if not (sec > 0):
            raise ValueError
    except Exception:  # noqa: BLE001
        return float(default)
    return max(min_s, min(max_s, sec))


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
    """Return an ephemeral free TCP port, then close the socket."""  # AI-AGENT-REF
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def get_pid_on_port(port: int) -> int | None:
    """Best-effort PID lookup for ``port``."""  # AI-AGENT-REF
    try:
        import psutil  # type: ignore

        for p in psutil.process_iter(attrs=["pid", "connections"]):
            for c in p.info.get("connections") or []:
                if getattr(c, "laddr", None) and getattr(c.laddr, "port", None) == port:
                    return p.info["pid"]
    except Exception:  # noqa: BLE001
        pass
    return None


def health_check(df, resolution: str) -> bool:
    """Minimal DataFrame health check with env override."""  # AI-AGENT-REF
    try:
        rows = int(os.getenv("HEALTH_MIN_ROWS", "100"))
    except Exception:  # noqa: BLE001
        rows = 100
    try:
        n = len(df) if df is not None else 0
    except Exception:  # noqa: BLE001
        n = 0
    return n >= rows


def sleep_s(seconds: float) -> None:
    """Thin wrapper so tests can monkeypatch easily."""  # AI-AGENT-REF
    time.sleep(clamp_timeout(seconds, default=0.01))


def sleep(seconds: float) -> None:
    """Backward compatible sleep wrapper."""  # AI-AGENT-REF
    sleep_s(seconds)


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
    "sleep_s",
    "sleep",
]
