from __future__ import annotations

import os
import time as _time

_real_sleep = _time.sleep
from typing import Optional, Union

HTTP_TIMEOUT: Union[int, float] = float(os.getenv("HTTP_TIMEOUT", "10"))  # AI-AGENT-REF: canonical timeout across runtime


def clamp_timeout(value: Optional[float]) -> float:
    """Return a sane timeout, falling back to HTTP_TIMEOUT when None/invalid."""  # AI-AGENT-REF: clamp helper
    try:
        if value is None:
            return HTTP_TIMEOUT
        v = float(value)
        return v if v > 0 else HTTP_TIMEOUT
    except Exception:
        return HTTP_TIMEOUT


def _robust_sleep(seconds: Union[int, float]) -> None:
    """Block for ``seconds`` using the original ``time.sleep`` with a minimum delay."""  # AI-AGENT-REF: deterministic sleep

    try:
        s = float(seconds)
    except Exception:
        s = 0.0
    _real_sleep(max(s, 0.01))


_force_local_sleep = str(os.getenv("AI_TRADING_FORCE_LOCAL_SLEEP", "1")).lower() in {"1", "true", "yes", "on"}
if _force_local_sleep:
    sleep = _robust_sleep  # type: ignore[assignment]
else:  # pragma: no cover
    sleep = _time.sleep

__all__ = ["HTTP_TIMEOUT", "clamp_timeout", "sleep"]
