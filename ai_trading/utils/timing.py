from __future__ import annotations
import os
import time as _time
HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', os.getenv('HTTP_TIMEOUT_S', '10')))
SUBPROCESS_TIMEOUT_S = 5.0

def _is_test_env() -> bool:
    return os.getenv('PYTEST_RUNNING', '').lower() in {'1', 'true', 'yes'} or os.getenv('TESTING', '').lower().startswith('true')

def sleep(seconds: float) -> None:
    if _is_test_env():
        seconds = min(seconds or 0.0, 0.05)
    _time.sleep(max(0.0, seconds))

def clamp_timeout(value: float | None=None, *, default_non_test: float=0.75, default_test: float=0.25, min_s: float=0.05, max_s: float=15.0) -> float:
    test_env = _is_test_env()
    out = default_test if value is None and test_env else default_non_test if value is None else float(value)
    return max(min_s, min(max_s, out))
__all__ = ['sleep', 'clamp_timeout', 'HTTP_TIMEOUT', 'SUBPROCESS_TIMEOUT_S']