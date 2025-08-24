from __future__ import annotations

"""
ai_trading.utils (package)
Ensures deterministic, measurable sleep for tests and workers.
"""

import os as _os
from time import perf_counter as _pc, sleep as _os_sleep
from typing import Union as _Union

# Re-export existing timing utilities from the local timing module
try:
    from .timing import (  # type: ignore
        HTTP_TIMEOUT as HTTP_TIMEOUT,
        clamp_timeout as clamp_timeout,
        perf_counter as perf_counter,
        sleep as _orig_sleep,
    )
except Exception:  # pragma: no cover - very defensive
    HTTP_TIMEOUT = 10.0  # type: ignore[assignment]

    def clamp_timeout(x):  # type: ignore[no-redef]
        return float(HTTP_TIMEOUT) if (x is None or float(x) < 0) else float(x)

    perf_counter = _pc  # type: ignore[assignment]
    _orig_sleep = None  # type: ignore[assignment]


def _robust_sleep(seconds: _Union[int, float]) -> None:
    """Block for ~seconds ensuring measurable delay."""  # AI-AGENT-REF: deterministic sleep
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


_force_local = str(_os.getenv("AI_TRADING_FORCE_LOCAL_SLEEP", "1")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if _force_local or _orig_sleep is None:
    sleep = _robust_sleep  # type: ignore[assignment]
else:
    sleep = _orig_sleep  # type: ignore[assignment]


from .base import (  # noqa: E402
    EASTERN_TZ,
    ensure_utc,
    get_free_port,
    get_pid_on_port,
    health_check,
    is_market_open,
    market_open_between,
    # back-compat exports used by ai_trading.core.bot_engine
    log_warning,
    model_lock,
    safe_to_datetime,
    validate_ohlcv,
    # subprocess helpers for git hash retrieval
    SUBPROCESS_TIMEOUT_DEFAULT,
    safe_subprocess_run,
)

# Keep submodules importable as ai_trading.utils.http, etc.
from . import http  # noqa: F401

__all__ = [
    "HTTP_TIMEOUT",
    "clamp_timeout",
    "perf_counter",
    "sleep",
    "EASTERN_TZ",
    "ensure_utc",
    "get_free_port",
    "get_pid_on_port",
    "health_check",
    "is_market_open",
    "market_open_between",
    "http",
    # back-compat (engine)
    "log_warning",
    "model_lock",
    "safe_to_datetime",
    "validate_ohlcv",
    # subprocess helper
    "SUBPROCESS_TIMEOUT_DEFAULT",
    "safe_subprocess_run",
]

