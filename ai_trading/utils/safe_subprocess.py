"""Utilities for running subprocesses with safety guards."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Sequence
import shlex

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
    """Run ``cmd`` safely with timeout handling."""

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
        with subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=run_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    # Drain any remaining output buffers to avoid zombie processes.
                    proc.communicate()
                except Exception:
                    # Ignore secondary errors during cleanup.
                    pass
                return SafeSubprocessResult("", "", 124, True)
            stdout = stdout or ""
            stderr = stderr or ""
            returncode = proc.returncode
    except OSError as exc:
        logger.warning("safe_subprocess_run(%s) failed: %s", argv, exc)
        return SafeSubprocessResult("", str(exc), getattr(exc, "returncode", -1), False)
    except subprocess.SubprocessError as exc:
        # ``TimeoutExpired`` is handled above; this branch captures other subprocess failures.
        logger.warning("safe_subprocess_run(%s) failed: %s", argv, exc)
        return SafeSubprocessResult("", str(exc), getattr(exc, "returncode", -1), False)

    return SafeSubprocessResult(stdout, stderr, returncode, False)
