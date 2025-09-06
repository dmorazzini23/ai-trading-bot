from __future__ import annotations

import os
from typing import Optional, Union

from .sleep import sleep

# Prefer AI_HTTP_TIMEOUT when present (tests set this); fallback to HTTP_TIMEOUT env
HTTP_TIMEOUT: Union[int, float] = float(
    os.getenv("AI_HTTP_TIMEOUT", os.getenv("HTTP_TIMEOUT", "10"))
)  # AI-AGENT-REF: canonical timeout across runtime


def clamp_timeout(value: Optional[float]) -> float:
    """Return a sane timeout, falling back to HTTP_TIMEOUT when None/invalid."""  # AI-AGENT-REF: clamp helper
    try:
        if value is None:
            return HTTP_TIMEOUT
        v = float(value)
        return v if v > 0 else HTTP_TIMEOUT
    except (TypeError, ValueError):
        return HTTP_TIMEOUT


__all__ = ["HTTP_TIMEOUT", "clamp_timeout", "sleep"]
