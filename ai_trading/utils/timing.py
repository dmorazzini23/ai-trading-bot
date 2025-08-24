from __future__ import annotations

import os
from time import perf_counter as _pc, sleep as _os_sleep
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


def _robust_sleep(seconds: Union[int, float]) -> None:
    """Block for ~seconds ensuring measurable delay even if time.sleep is patched."""  # AI-AGENT-REF: deterministic sleep
    s = float(seconds)
    if s <= 0.0:
        return
    start = _pc()
    spin_cap = min(0.050, s)
    while True:
        if (_pc() - start) >= spin_cap:
            break
    remaining = s - (_pc() - start)
    if remaining > 0.0:
        _os_sleep(remaining)
    while (_pc() - start) < s:
        pass


_force_local_sleep = str(os.getenv("AI_TRADING_FORCE_LOCAL_SLEEP", "1")).lower() in {"1", "true", "yes", "on"}
if _force_local_sleep:
    sleep = _robust_sleep  # type: ignore[assignment]
else:  # pragma: no cover
    sleep = _os_sleep

__all__ = ["HTTP_TIMEOUT", "clamp_timeout", "sleep"]
