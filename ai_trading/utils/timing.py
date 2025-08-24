from __future__ import annotations

import time
from typing import Optional

HTTP_TIMEOUT: float = 10.0  # default for HTTP operations  # AI-AGENT-REF: canonical timeout


def clamp_timeout(value: Optional[float]) -> float:
    """Return a sane timeout; fall back to HTTP_TIMEOUT if None/invalid."""  # AI-AGENT-REF: clarified doc
    try:
        if value is None:
            return HTTP_TIMEOUT
        v = float(value)
        return v if v > 0 else HTTP_TIMEOUT
    except Exception:
        return HTTP_TIMEOUT


def sleep(seconds: float) -> None:
    """Small wrapper to keep sleeps centralized/testable."""  # AI-AGENT-REF: unified sleep helper
    time.sleep(float(seconds))
