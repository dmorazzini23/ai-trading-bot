"""Utilities for running subprocesses with safety guards."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Sequence

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
    timeout: float | int | None = None,
    **popen_kwargs,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` with a defensive timeout and structured logging."""

    argv = _coerce_cmd(cmd)
    popen_args = dict(popen_kwargs)

    raw_timeout = popen_args.pop("timeout", timeout)
    run_timeout = SUBPROCESS_TIMEOUT_S if raw_timeout is None else float(raw_timeout)
    if run_timeout <= 0:
        run_timeout = SUBPROCESS_TIMEOUT_S

    capture_output = popen_args.get("capture_output")
    if not capture_output:
        popen_args.setdefault("stdout", subprocess.PIPE)
        popen_args.setdefault("stderr", subprocess.PIPE)
    popen_args.setdefault("text", True)
    popen_args["timeout"] = run_timeout

    try:
        completed = subprocess.run(argv, **popen_args)
    except subprocess.TimeoutExpired as exc:
        stdout_text = _normalize_stream(getattr(exc, "output", None))
        stderr_text = _normalize_stream(getattr(exc, "stderr", None))
        logger.warning(
            "SAFE_SUBPROCESS_TIMEOUT",
            extra={
                "cmd": argv,
                "timeout": run_timeout,
                "stdout": stdout_text,
                "stderr": stderr_text,
            },
        )
        _prepare_timeout_exception(exc, run_timeout)
        raise

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
