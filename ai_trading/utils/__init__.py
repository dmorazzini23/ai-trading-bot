from __future__ import annotations
import time as _time

# timing helpers for public surface  # AI-AGENT-REF: re-export
from .timing import HTTP_TIMEOUT, clamp_timeout

def sleep(seconds: float) -> None:
    """Synchronous sleep (non-negative) using the real time.sleep.

    We bind directly to time.sleep to remove any ambiguity from intermediate
    wrappers or lazy attribute resolution.
    """
    _time.sleep(max(0.0, float(seconds)))  # AI-AGENT-REF: direct time.sleep call
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
