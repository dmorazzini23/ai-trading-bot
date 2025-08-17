from __future__ import annotations

import os
import time as _time

# AI-AGENT-REF: centralized timeout constants
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))
SUBPROCESS_TIMEOUT_S = 5.0


def _is_test_env() -> bool:
    return any(
        os.getenv(k, "").lower() in {"1", "true", "yes"}
        for k in ("PYTEST_RUNNING", "TESTING")
    )


# default clamp used in *all* retry/sleep loops under tests
def sleep(seconds: float) -> None:
    if _is_test_env():
        seconds = min(seconds, 0.05)  # cap sleeps in tests
    _time.sleep(max(0.0, seconds))


def clamp_timeout(
    value: float | None = None,
    *,
    default_non_test: float = 0.75,
    default_test: float = 0.25,
    min_s: float = 0.05,
    max_s: float = 15.0,
) -> float:
    """Return a timeout value respecting tests and bounds."""  # AI-AGENT-REF
    if value is None:
        value = default_test if _is_test_env() else default_non_test
    out = float(value)
    if out < min_s:
        out = min_s
    if out > max_s:
        out = max_s
    return out


__all__ = ["sleep", "clamp_timeout", "HTTP_TIMEOUT", "SUBPROCESS_TIMEOUT_S"]
