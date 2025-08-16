# ruff: noqa
from __future__ import annotations

"""Utility functions and helpers for the AI trading bot.

Unified utilities export layer.
Only re-export light symbols needed by production modules to avoid import-time heaviness.
"""
import os
from typing import Any

import pandas as pd

from .base import (
    EASTERN_TZ,
    HAS_PANDAS,
    ensure_utc,
    ensure_utc_index,
    get_free_port,
    get_latest_close,
    get_ohlcv_columns,
    get_pid_on_port,
    health_check,
    is_market_holiday,
    is_market_open,
    is_weekend,
    log_health_row_check,
    log_warning,
    model_lock,
    portfolio_lock,
    requires_pandas,
    safe_to_datetime,
    validate_ohlcv,
    validate_ohlcv_basic,
)
from .determinism import (
    ensure_deterministic_training,
    get_model_spec,
    lock_model_spec,
    set_random_seeds,
    unlock_model_spec,
)
from .process_manager import acquire_lock, file_lock, release_lock
from .time import now_utc
from .timing import sleep  # AI-AGENT-REF: test-aware timing helpers

HTTP_TIMEOUT_S: float = float(os.getenv("HTTP_TIMEOUT_S", "10") or 10)


def clamp_timeout(
    kwargs: dict[str, Any], default: float | None = None
) -> dict[str, Any]:
    """Ensure a numeric timeout is present in kwargs."""
    if default is None:
        default = HTTP_TIMEOUT_S
    t = kwargs.get("timeout", default)
    try:
        t = float(t)
    except Exception:
        t = default
    kwargs["timeout"] = t
    return kwargs


import ai_trading.utils.process_manager as process_manager  # noqa: E402

__all__ = [
    "log_warning",
    "model_lock",
    "safe_to_datetime",
    "validate_ohlcv",
    "validate_ohlcv_basic",
    "portfolio_lock",
    "is_market_open",
    "is_weekend",
    "is_market_holiday",
    "get_free_port",
    "get_pid_on_port",
    "log_health_row_check",
    "pd",
    "HAS_PANDAS",
    "requires_pandas",
    "set_random_seeds",
    "ensure_deterministic_training",
    "get_model_spec",
    "lock_model_spec",
    "unlock_model_spec",
    "now_utc",
    "sleep",
    "clamp_timeout",
    "acquire_lock",
    "release_lock",
    "file_lock",
    "get_latest_close",
    "EASTERN_TZ",
    "health_check",
    "ensure_utc",
    "get_ohlcv_columns",
    "ensure_utc_index",
    "process_manager",
    "HTTP_TIMEOUT_S",
]
