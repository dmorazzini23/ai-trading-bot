"""Utilities for running subprocesses with safety guards."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence

from ai_trading.logging import get_logger

logger = get_logger(__name__)

# Default timeout for subprocess operations in seconds
SUBPROCESS_TIMEOUT_S = 0.3

# Back-compat alias expected by callers
SUBPROCESS_TIMEOUT_DEFAULT = SUBPROCESS_TIMEOUT_S


@dataclass(slots=True)
class SafeSubprocessResult:
    """Lightweight result container for ``safe_subprocess_run`` timeouts."""

    stdout: str
    stderr: str
    returncode: int
    timeout: bool


def _coerce_cmd(cmd: Sequence[str] | str) -> list[str]:
    if isinstance(cmd, (list, tuple)):
        return [str(part) for part in cmd]
    if isinstance(cmd, str):
        return shlex.split(cmd)
    raise TypeError("cmd must be a sequence or string")


def safe_subprocess_run(
    cmd: Sequence[str] | str,
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` with a defensive timeout and structured logging."""

    argv = _coerce_cmd(cmd)
    resolved_timeout = SUBPROCESS_TIMEOUT_S if timeout is None else float(timeout)
    if resolved_timeout <= 0:
        resolved_timeout = SUBPROCESS_TIMEOUT_S

    run_kwargs = {
        "check": False,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "timeout": resolved_timeout,
        "env": dict(env) if env is not None else None,
    }

    try:
        completed = subprocess.run(argv, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        _prepare_timeout_exception(exc, resolved_timeout)
        logger.warning(
            "SAFE_SUBPROCESS_TIMEOUT",
            extra={
                "cmd": argv,
                "timeout": resolved_timeout,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            },
        )
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        completed = _coerce_exception_result(exc, argv)
        logger.warning(
            "SAFE_SUBPROCESS_ERROR",
            extra={
                "cmd": argv,
                "returncode": completed.returncode,
                "error": str(exc),
                "exc_type": type(exc).__name__,
            },
        )
        return completed

    completed.stdout = _normalize_stream(completed.stdout)
    completed.stderr = _normalize_stream(completed.stderr)
    return completed


def _normalize_stream(stream: str | bytes | None) -> str:
    """Return a safe string representation of subprocess output."""

    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def _prepare_timeout_exception(exc: subprocess.TimeoutExpired, run_timeout: float) -> None:
    """Attach ``SafeSubprocessResult`` metadata to ``TimeoutExpired``."""

    stdout_text = _normalize_stream(getattr(exc, "output", None))
    stderr_text = _normalize_stream(getattr(exc, "stderr", None))
    result = SafeSubprocessResult(stdout_text, stderr_text, 124, True)
    exc.timeout = run_timeout
    exc.stdout = stdout_text
    exc.stderr = stderr_text
    exc.result = result


def _coerce_exception_result(
    exc: OSError | subprocess.SubprocessError, argv: Sequence[str] | str
) -> subprocess.CompletedProcess[str]:
    """Return a ``CompletedProcess`` representing an execution failure."""

    if isinstance(exc, subprocess.CalledProcessError):
        stdout_text = _normalize_stream(getattr(exc, "stdout", None))
        stderr_text = _normalize_stream(getattr(exc, "stderr", None))
        cmd = exc.cmd if exc.cmd is not None else argv
        return subprocess.CompletedProcess(cmd, exc.returncode, stdout_text, stderr_text)

    stdout_text = _normalize_stream(getattr(exc, "stdout", None))
    stderr_value = getattr(exc, "stderr", None)
    stderr_text = _normalize_stream(stderr_value) if stderr_value is not None else str(exc)

    returncode = getattr(exc, "returncode", None)
    if returncode is None:
        if isinstance(exc, FileNotFoundError):
            returncode = 127
        elif isinstance(exc, OSError) and getattr(exc, "errno", None):
            returncode = exc.errno or 1
        else:
            returncode = 1

    return subprocess.CompletedProcess(argv, returncode, stdout_text, stderr_text)
