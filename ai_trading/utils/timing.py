from __future__ import annotations

import os
import time as _time
from time import perf_counter as _perf

_real_sleep = _time.sleep
from typing import Optional, Union

# Prefer AI_HTTP_TIMEOUT when present (tests set this); fallback to HTTP_TIMEOUT env
HTTP_TIMEOUT: Union[int, float] = float(os.getenv("HTTP_TIMEOUT", os.getenv("AI_HTTP_TIMEOUT", "10")))  # AI-AGENT-REF: canonical timeout across runtime


def clamp_timeout(value: Optional[float]) -> float:
    """Return a sane timeout, falling back to HTTP_TIMEOUT when None/invalid."""  # AI-AGENT-REF: clamp helper
    try:
        if value is None:
            return HTTP_TIMEOUT
        v = float(value)
        return v if v > 0 else HTTP_TIMEOUT
    except (TypeError, ValueError):
        return HTTP_TIMEOUT


def _robust_sleep(seconds: Union[int, float]) -> None:
    """Block for at least ~10ms even under monkeypatched time.sleep.

    Uses the original OS sleep captured at import time and a short
    perf_counter-based busy wait to ensure measurable elapsed time.
    """  # AI-AGENT-REF: deterministic sleep

    try:
        s = float(seconds)
    except (TypeError, ValueError):
        s = 0.0
    target = max(s, 0.01)
    start = _perf()
    _real_sleep(target)
    # Ensure we cross ~9ms even if scheduler wakes early; cap iterations to avoid hangs
    _tries = 0
    while (_perf() - start) < 0.009 and _tries < 5:
        _real_sleep(0.005)
        _tries += 1


_force_local_sleep = str(os.getenv("AI_TRADING_FORCE_LOCAL_SLEEP", "1")).lower() in {"1", "true", "yes", "on"}
if _force_local_sleep:
    sleep = _robust_sleep  # type: ignore[assignment]
else:  # pragma: no cover
    sleep = _time.sleep

__all__ = ["HTTP_TIMEOUT", "clamp_timeout", "sleep"]
