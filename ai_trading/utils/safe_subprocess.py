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
    t = float(timeout) if timeout is not None else SUBPROCESS_TIMEOUT_DEFAULT
    try:
        res = subprocess.run(list(cmd), timeout=t, check=True, capture_output=True)
    except subprocess.TimeoutExpired:
        logger.warning("safe_subprocess_run(%s) timed out after %.2f seconds", cmd, t)
        return SafeSubprocessResult(stdout="", returncode=-1, timeout=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("safe_subprocess_run(%s) failed: %s", cmd, exc)
        return SafeSubprocessResult(stdout="", returncode=exc.returncode, timeout=False)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("safe_subprocess_run(%s) failed: %s", cmd, exc)
        return SafeSubprocessResult(stdout="", returncode=getattr(exc, "returncode", -1))
    stdout = (res.stdout or b"").decode(errors="ignore").strip()
    return SafeSubprocessResult(stdout=stdout, returncode=res.returncode)
