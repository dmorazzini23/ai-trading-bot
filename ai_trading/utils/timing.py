from __future__ import annotations

import time
from typing import Optional, Union

HTTP_TIMEOUT: Union[int, float] = 10.0  # AI-AGENT-REF: canonical timeout across runtime


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
    """Real sleep (non-negative). Thin wrapper to allow test monkeypatching."""
    time.sleep(max(0.0, float(seconds)))
