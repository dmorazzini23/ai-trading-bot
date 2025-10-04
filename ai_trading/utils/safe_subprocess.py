"""Utilities for running subprocesses with safety guards."""

from __future__ import annotations

import shlex
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from typing import Sequence

from ai_trading.logging import get_logger

logger = get_logger(__name__)

# Default timeout for subprocess operations in seconds
SUBPROCESS_TIMEOUT_S = 5.0

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
    timeout: float | int | None = None,
) -> SafeSubprocessResult:
    """Run ``cmd`` safely with timeout handling.

    Raises
    ------
    subprocess.TimeoutExpired
        If ``cmd`` exceeds ``timeout``. The raised exception exposes the
        ``SafeSubprocessResult`` via ``exc.result`` with ``timeout=True`` and a
        ``124`` return code.
    """

    run_timeout = SUBPROCESS_TIMEOUT_DEFAULT if timeout is None else float(timeout)
    if run_timeout <= 0:
        logger.warning(
            "safe_subprocess_run(%s) timed out immediately (timeout=%.2f seconds)",
            cmd,
            run_timeout,
        )
        return SafeSubprocessResult("", "", -1, True)

    argv = _coerce_cmd(cmd)
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        logger.warning("safe_subprocess_run(%s) failed: %s", argv, exc)
        return SafeSubprocessResult("", str(exc), getattr(exc, "returncode", -1), False)

    try:
        stdout_text, stderr_text = proc.communicate(timeout=run_timeout)
    except subprocess.TimeoutExpired as exc:
        exc.timeout = run_timeout
        _augment_timeout_exception(proc, exc)
        raise
    except subprocess.SubprocessError as exc:
        with suppress(ProcessLookupError):
            proc.kill()
        proc.wait()
        logger.warning("safe_subprocess_run(%s) failed: %s", argv, exc)
        return SafeSubprocessResult("", str(exc), getattr(exc, "returncode", -1), False)

    stdout_text = _normalize_stream(stdout_text)
    stderr_text = _normalize_stream(stderr_text)
    return SafeSubprocessResult(
        stdout_text,
        stderr_text,
        proc.returncode if proc.returncode is not None else 0,
        False,
    )


def _normalize_stream(stream: str | bytes | None) -> str:
    """Return a safe string representation of subprocess output."""

    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def _augment_timeout_exception(
    proc: subprocess.Popen[bytes] | subprocess.Popen[str],
    exc: subprocess.TimeoutExpired,
) -> None:
    """Populate timeout exception metadata before re-raising."""

    with suppress(ProcessLookupError):
        proc.kill()
    stdout_after, stderr_after = proc.communicate()

    def _merge_streams(primary: str | bytes | None, secondary: str | bytes | None) -> str:
        first = _normalize_stream(primary)
        second = _normalize_stream(secondary)
        if first and second:
            if second.startswith(first):
                return second
            if first.endswith(second):
                return first
            return first + second
        return first or second

    collected_stdout = _merge_streams(getattr(exc, "stdout", None), stdout_after)
    collected_stderr = _merge_streams(getattr(exc, "stderr", None), stderr_after)

    result = SafeSubprocessResult(collected_stdout, collected_stderr, 124, True)
    proc.returncode = 124
    exc.stdout = collected_stdout
    exc.stderr = collected_stderr
    exc.result = result
