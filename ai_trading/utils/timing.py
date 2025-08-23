from __future__ import annotations

import time
from typing import Optional

# Default HTTP timeout used by runtime unless overridden
HTTP_TIMEOUT: float = 10.0


def clamp_timeout(value: Optional[float]) -> float:
    """Return a sane timeout, falling back to HTTP_TIMEOUT when None/invalid."""
    try:
        if value is None:
            return HTTP_TIMEOUT
        v = float(value)
        return v if v > 0 else HTTP_TIMEOUT
    except Exception:
        return HTTP_TIMEOUT


def sleep(seconds: float) -> None:
    """Small wrapper for testability and central control."""
    time.sleep(float(seconds))
