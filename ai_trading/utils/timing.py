from __future__ import annotations
import os
import time
from typing import Optional

# Default HTTP timeout (seconds). Overridable via env for ops.
HTTP_TIMEOUT: float = float(os.getenv("AI_TRADING_HTTP_TIMEOUT", "10"))

def clamp_timeout(value: Optional[float], *, min_s: float = 0.1, max_s: float = 120.0) -> float:
    """Normalize a caller-provided timeout into a safe range."""
    if value is None:
        return HTTP_TIMEOUT
    try:
        v = float(value)
    except Exception:
        return HTTP_TIMEOUT
    if v < min_s:
        return min_s
    if v > max_s:
        return max_s
    return v

def sleep(seconds: float) -> None:
    """Simple blocking sleep used by backoff/retry paths."""
    time.sleep(max(0.0, float(seconds)))
