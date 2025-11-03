# ai_trading/utils/safe_subprocess.py
from __future__ import annotations

import logging
import math
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

    requested_check = popen_kwargs.pop("check", check)
    text_mode = popen_kwargs.pop("text", text)

    timeout_param: float | None
    if timeout is None:
        timeout_param = SUBPROCESS_TIMEOUT_DEFAULT
    else:
        try:
            normalized_timeout = float(timeout)
        except (TypeError, ValueError):
            normalized_timeout = SUBPROCESS_TIMEOUT_DEFAULT
        if not math.isfinite(normalized_timeout):
            timeout_param = SUBPROCESS_TIMEOUT_DEFAULT
        elif normalized_timeout <= 0:
            timeout_param = None
        else:
            timeout_param = normalized_timeout

    run_kwargs = dict(popen_kwargs)
    run_kwargs.pop("timeout", None)
    run_kwargs["check"] = False
    run_kwargs["text"] = text_mode

    if capture_output and "stdout" not in run_kwargs and "stderr" not in run_kwargs:
        run_kwargs["stdout"] = subprocess.PIPE
        run_kwargs["stderr"] = subprocess.PIPE
    if timeout_param is not None:
        run_kwargs["timeout"] = timeout_param

    try:
        completed = subprocess.run(cmd, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        stdout_text = _normalize_stream(getattr(exc, "output", None))
        stderr_text = _normalize_stream(getattr(exc, "stderr", None))
        result = SafeSubprocessResult(stdout_text, stderr_text, 124, True)
        exc.stdout = stdout_text
        exc.stderr = stderr_text
        exc.result = result
        exc.timeout = timeout_param
        log.warning(
            "SAFE_SUBPROCESS_TIMEOUT",
            extra={"cmd": cmd, "timeout": timeout_param},
        )
        raise exc
    except (OSError, subprocess.SubprocessError) as exc:
        result = _coerce_exception_result(exc, cmd)
        log.warning(
            "SAFE_SUBPROCESS_ERROR",
            extra={"cmd": cmd, "returncode": result.returncode, "error": str(exc), "exc_type": type(exc).__name__},
        )
        return result

    stdout_text = _normalize_stream(completed.stdout)
    stderr_text = _normalize_stream(completed.stderr)
    ret = subprocess.CompletedProcess(cmd, completed.returncode, stdout_text, stderr_text)
    if requested_check and ret.returncode != 0:
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
