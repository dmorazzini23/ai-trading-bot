"""Lightweight re-exports for utility helpers.

This module avoids importing heavy dependencies at package import time by
deferring optional components to ``__getattr__``.
"""

from __future__ import annotations

from .timing import HTTP_TIMEOUT, clamp_timeout, sleep  # AI-AGENT-REF: re-export timing
from .optdeps import OptionalDependencyError, optional_import, module_ok  # AI-AGENT-REF: re-export opt deps

_BASE_EXPORTS = {
    "EASTERN_TZ",
    "ensure_utc",
    "get_free_port",
    "get_pid_on_port",
    "health_check",
    "is_market_open",
    "market_open_between",
    "log_warning",
    "model_lock",
    "safe_to_datetime",
    "validate_ohlcv",
    "SUBPROCESS_TIMEOUT_DEFAULT",
    "safe_subprocess_run",
}

__all__ = [
    "HTTP_TIMEOUT", "clamp_timeout", "sleep",
    "OptionalDependencyError", "optional_import", "module_ok",
    *sorted(_BASE_EXPORTS),
    "http",
]


def __getattr__(name: str):  # AI-AGENT-REF: lazy base/http exports
    if name == "http":
        from . import http
        return http
    if name in _BASE_EXPORTS:
        from .base import (
            EASTERN_TZ,
            ensure_utc,
            get_free_port,
            get_pid_on_port,
            health_check,
            is_market_open,
            market_open_between,
            log_warning,
            model_lock,
            safe_to_datetime,
            validate_ohlcv,
            SUBPROCESS_TIMEOUT_DEFAULT,
            safe_subprocess_run,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")

