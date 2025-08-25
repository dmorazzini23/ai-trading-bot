from __future__ import annotations

"""
ai_trading.utils (package)
Ensures deterministic, measurable sleep for tests and workers.
"""

import os
import time
from typing import Callable

# Re-export timing helpers from the local module
try:
    from .timing import (  # type: ignore
        HTTP_TIMEOUT as HTTP_TIMEOUT,
        clamp_timeout as clamp_timeout,
    )
except Exception:  # pragma: no cover - very defensive
    HTTP_TIMEOUT = 10.0  # type: ignore[assignment]

    def clamp_timeout(x):  # type: ignore[no-redef]
        return float(HTTP_TIMEOUT) if (x is None or float(x) < 0) else float(x)


# Re-export stdlib perf_counter EXACTLY (tests rely on this identity)
perf_counter = time.perf_counter

# Optional fast path to OS sleep for very large waits
_os_sleep: Callable[[float], None] = time.sleep


def _robust_sleep(seconds: float) -> None:
    """Sleep for ~seconds with a deterministic, measurable delay."""  # AI-AGENT-REF: fix perf_counter export and robust sleep
    if not seconds or seconds <= 0.0:
        return

    deadline = perf_counter() + float(seconds)
    while True:
        now = perf_counter()
        remaining = deadline - now
        if remaining <= 0.0:
            break
        if remaining > 0.02:
            # sleep in small chunks to avoid overshoot on coarse timers
            _os_sleep(0.01 if remaining > 0.05 else 0.005)
        else:
            while perf_counter() < deadline:
                pass
            break


# Allow opting out of robust sleep (use raw OS sleep only) via env
_FORCE_OS_SLEEP = os.getenv("AI_TRADING_FORCE_LOCAL_SLEEP") in {"1", "true", "True", "yes"}

# Exported public sleep: direct impl (no wrapper indirection)
sleep: Callable[[float], None] = _os_sleep if _FORCE_OS_SLEEP else _robust_sleep


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
# AI-AGENT-REF: expose optional dependency helpers
from .optdeps import optional_import, module_ok  # noqa: F401

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
    "optional_import",
    "module_ok",
]

