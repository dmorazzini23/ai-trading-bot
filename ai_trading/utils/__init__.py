from __future__ import annotations

# timing helpers for public surface  # AI-AGENT-REF: re-export
from .timing import HTTP_TIMEOUT, clamp_timeout, sleep  # AI-AGENT-REF: robust sleep
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
