# ai_trading/utils/safe_subprocess.py
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SafeSubprocessResult:
    stdout: str
    stderr: str
    returncode: int
    timeout: bool


SUBPROCESS_TIMEOUT_S = 0.3
SUBPROCESS_TIMEOUT_DEFAULT = SUBPROCESS_TIMEOUT_S


def safe_subprocess_run(
    cmd: Sequence[str],
    *,
    timeout: Optional[float] = None,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    **popen_kwargs,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with a hard timeout that surfaces subprocess.TimeoutExpired
    to callers (tests expect this). Never swallow TimeoutExpired.

    - If timeout is given and the child hangs, kill it, read remaining IO,
      log a WARNING, and re-raise TimeoutExpired.
    - Mirrors subprocess.run behavior for 'check', 'capture_output', 'text'.
    """

    if capture_output:
        popen_kwargs.setdefault("stdout", subprocess.PIPE)
        popen_kwargs.setdefault("stderr", subprocess.PIPE)

    resolved_timeout = None
    if timeout is not None and timeout > 0:
        resolved_timeout = timeout

    run_kwargs = dict(popen_kwargs)
    run_kwargs.setdefault("check", False)
    run_kwargs.setdefault("text", text)
    run_kwargs["timeout"] = resolved_timeout

    try:
        completed = subprocess.run(cmd, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        stdout = _normalize_stream(getattr(exc, "output", None))
        stderr = _normalize_stream(getattr(exc, "stderr", None))
        log.warning("SAFE_SUBPROCESS_TIMEOUT", extra={"cmd": cmd, "timeout": timeout if timeout is not None else resolved_timeout})
        result = SafeSubprocessResult(stdout, stderr, 124, True)
        exc.stdout = stdout
        exc.stderr = stderr
        exc.result = result
        if timeout is not None:
            exc.timeout = timeout
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        result = _coerce_exception_result(exc, cmd)
        log.warning(
            "SAFE_SUBPROCESS_ERROR",
            extra={"cmd": cmd, "returncode": result.returncode, "error": str(exc), "exc_type": type(exc).__name__},
        )
        return result

    stdout = _normalize_stream(completed.stdout)
    stderr = _normalize_stream(completed.stderr)
    ret = subprocess.CompletedProcess(cmd, completed.returncode, stdout, stderr)
    if check and ret.returncode != 0:
        raise subprocess.CalledProcessError(ret.returncode, cmd, ret.stdout, ret.stderr)
    return ret


def _normalize_stream(stream: Optional[str | bytes]) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream


def _coerce_exception_result(exc: OSError | subprocess.SubprocessError, cmd: Sequence[str]) -> subprocess.CompletedProcess:
    stdout = _normalize_stream(getattr(exc, "stdout", None))
    stderr_value = getattr(exc, "stderr", None)
    if stderr_value is None:
        stderr = str(exc)
    else:
        stderr = _normalize_stream(stderr_value)

    returncode = getattr(exc, "returncode", None)
    if returncode is None:
        if isinstance(exc, FileNotFoundError):
            returncode = 127
        elif isinstance(exc, OSError) and getattr(exc, "errno", None):
            returncode = exc.errno or 1
        else:
            returncode = 1

    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
