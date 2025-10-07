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
    """Lightweight result container for ``safe_subprocess_run``."""

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
) -> SafeSubprocessResult:
    """Run ``cmd`` safely with timeout handling.

    Raises
    ------
    subprocess.TimeoutExpired
        If ``cmd`` exceeds ``timeout``. The raised exception exposes the
        ``SafeSubprocessResult`` via ``exc.result`` with ``timeout=True`` and a
        ``124`` return code.
    """

    run_timeout = (
        SUBPROCESS_TIMEOUT_S if timeout is None else max(0.0, float(timeout))
    )
    if run_timeout <= 0:
        logger.warning(
            "safe_subprocess_run(%s) timed out immediately (timeout=%.2f seconds)",
            cmd,
            run_timeout,
        )
        return SafeSubprocessResult("", "", -1, True)

    argv = _coerce_cmd(cmd)
    popen_args = dict(popen_kwargs)
    capture_output = popen_args.get("capture_output")
    if not capture_output:
        popen_args.setdefault("stdout", subprocess.PIPE)
        popen_args.setdefault("stderr", subprocess.PIPE)
    popen_args.setdefault("text", True)
    popen_args["timeout"] = run_timeout
    popen_args["check"] = False

    try:
        completed = subprocess.run(
            argv,
            **popen_args,
        )
    except subprocess.TimeoutExpired as exc:
        _prepare_timeout_exception(exc, run_timeout)
        raise
    except OSError as exc:
        logger.warning("safe_subprocess_run(%s) failed: %s", argv, exc)
        return SafeSubprocessResult("", str(exc), getattr(exc, "returncode", -1), False)
    except subprocess.SubprocessError as exc:
        logger.warning("safe_subprocess_run(%s) failed: %s", argv, exc)
        return SafeSubprocessResult("", str(exc), getattr(exc, "returncode", -1), False)

    stdout_text = _normalize_stream(completed.stdout)
    stderr_text = _normalize_stream(completed.stderr)
    return SafeSubprocessResult(
        stdout_text,
        stderr_text,
        completed.returncode if completed.returncode is not None else 0,
        False,
    )


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
