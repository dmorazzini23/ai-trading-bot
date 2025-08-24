from __future__ import annotations

import time
from typing import Optional

HTTP_TIMEOUT: float = 10.0  # AI-AGENT-REF: canonical timeout across runtime


def clamp_timeout(value: Optional[float]) -> float:
    """Return a sane timeout, falling back to HTTP_TIMEOUT when None/invalid."""  # AI-AGENT-REF: clamp helper
    try:
        if value is None:
            return HTTP_TIMEOUT
        v = float(value)
        return v if v > 0 else HTTP_TIMEOUT
    except Exception:
        return HTTP_TIMEOUT


def sleep(seconds: float) -> None:
    """Small wrapper for testability and centralized control."""  # AI-AGENT-REF: sleep wrapper
    time.sleep(float(seconds))
