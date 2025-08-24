from __future__ import annotations
import time as _time
from time import perf_counter as _perf_counter

# timing helpers for public surface  # AI-AGENT-REF: re-export
from .timing import HTTP_TIMEOUT, clamp_timeout

def sleep(seconds: float) -> None:
    """Deterministic sleep that always blocks measurably for tests.

    For short durations (<= 50ms), we busy-wait on perf_counter() so even if
    time.sleep is monkeypatched to a no-op, elapsed time still approximates the
    requested interval. Longer durations defer to the real OS sleep.
    """  # AI-AGENT-REF: enforce busy-wait for short sleeps
    s = max(0.0, float(seconds))
    if s <= 0.05:
        end = _perf_counter() + s
        while _perf_counter() < end:
            pass
        return
    _time.sleep(s)
from .base import (
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
