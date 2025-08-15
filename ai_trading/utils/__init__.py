from __future__ import annotations

"""Utility functions and helpers for the AI trading bot."""

"""
Unified utilities export layer.
Only re-export light symbols needed by production modules to avoid import-time heaviness.
"""
from zoneinfo import ZoneInfo
import pandas as pd
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
        portfolio_lock,
        requires_pandas,
        safe_to_datetime,
        validate_ohlcv,
        validate_ohlcv_basic,
    )
except Exception:  # pragma: no cover - optional deps
    HAS_PANDAS = False
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

EASTERN_TZ = ZoneInfo("America/New_York")


def health_check(value: float) -> bool:
    try:
        return float(value) > 0.0
    except Exception:
        return False


def get_latest_close(df: pd.DataFrame | None) -> float:
    if df is None or df.empty:
        return 0.0
    for col in ("close", "Close", "adj_close", "Adj Close"):
        if col in df.columns:
            v = pd.to_numeric(df[col].iloc[-1], errors="coerce")
            try:
                x = float(v)
                return 0.0 if pd.isna(x) else x
            except Exception:
                continue
    return 0.0

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
    "get_latest_close",
    "EASTERN_TZ",
    "health_check",
]
