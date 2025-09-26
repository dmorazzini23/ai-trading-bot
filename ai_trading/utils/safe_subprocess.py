"""Utilities for running subprocesses with safety guards."""

from __future__ import annotations

import subprocess
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
    returncode: int
    timeout: bool = False


def safe_subprocess_run(
    cmd: Sequence[str], timeout: float | int | None = None
) -> SafeSubprocessResult:
    """Run ``cmd`` with ``subprocess.run`` and capture stdout text.

    Any errors are swallowed and emitted as warnings so callers can degrade
    gracefully. When a timeout occurs a dedicated warning message is emitted
    before returning an empty string and marking the result as timed out.
    """
    if timeout is None:
        timeout_for_run: float | int = SUBPROCESS_TIMEOUT_DEFAULT
        timeout_for_log = float(SUBPROCESS_TIMEOUT_DEFAULT)
    else:
        timeout_for_run = timeout
        timeout_for_log = float(timeout)
        if timeout_for_log <= 0:
            logger.warning(
                "safe_subprocess_run(%s) timed out immediately (timeout=%.2f seconds)",
                cmd,
                timeout_for_log,
            )
            return SafeSubprocessResult(stdout="", returncode=-1, timeout=True)
    try:
        res = subprocess.run(
            list(cmd), timeout=timeout_for_run, check=False, capture_output=True
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "safe_subprocess_run(%s) timed out after %.2f seconds", cmd, timeout_for_log
        )
        return SafeSubprocessResult(stdout="", returncode=-1, timeout=True)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("safe_subprocess_run(%s) failed: %s", cmd, exc)
        return SafeSubprocessResult(stdout="", returncode=getattr(exc, "returncode", -1))
    if res.returncode != 0:
        logger.warning(
            "safe_subprocess_run(%s) failed: return code %s", cmd, res.returncode
        )
        return SafeSubprocessResult(stdout="", returncode=res.returncode, timeout=False)
    stdout = (res.stdout or b"").decode(errors="ignore").strip()
    return SafeSubprocessResult(stdout=stdout, returncode=res.returncode)
