from __future__ import annotations

# timing helpers for public surface  # AI-AGENT-REF: re-export
from .timing import HTTP_TIMEOUT, clamp_timeout, sleep as _timing_sleep

def sleep(seconds: float) -> None:
    """Synchronous sleep (non-negative), guaranteed to call timing.sleep.

    We intentionally wrap instead of aliasing so that any prior alias/stub in
    this module cannot shadow the real implementation.
    """
    # AI-AGENT-REF: hard bind sleep to timing.sleep
    _timing_sleep(seconds)
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
