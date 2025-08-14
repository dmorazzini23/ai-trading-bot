"""Utility functions and helpers for the AI trading bot."""

"""
Unified utilities export layer.
Only re-export light symbols needed by production modules to avoid import-time heaviness.
"""
# Keep this import small; base.py already imports get_settings safely.
# AI-AGENT-REF: tolerate missing heavy dependencies
try:
    from .base import (
        HAS_PANDAS,
        get_free_port,
        get_pid_on_port,
        is_market_holiday,
        is_market_open,
        is_weekend,
        log_health_row_check,
        log_warning,
        model_lock,
        pd,
        portfolio_lock,
        requires_pandas,
        safe_to_datetime,
        validate_ohlcv,
        validate_ohlcv_basic,
    )
except Exception:  # pragma: no cover - optional deps
    HAS_PANDAS = False
    pd = None
    def _stub(*args, **kwargs):
        return None
    get_free_port = get_pid_on_port = is_market_holiday = is_market_open = is_weekend = _stub
    log_health_row_check = log_warning = model_lock = portfolio_lock = requires_pandas = _stub
    safe_to_datetime = validate_ohlcv = validate_ohlcv_basic = _stub
from .determinism import (
    ensure_deterministic_training,
    get_model_spec,
    lock_model_spec,
    set_random_seeds,
    unlock_model_spec,
)
from .time import now_utc
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
]
