"""Test-aware timing utilities."""

from __future__ import annotations

import os
import time as _time


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


def clamp_timeout(t: float | None, default_non_test: float, default_test: float) -> float:
    """Return a timeout value respecting tests."""
    if _is_test_env():
        return default_test
    return default_non_test if t is None else t


__all__ = ["sleep", "clamp_timeout"]

