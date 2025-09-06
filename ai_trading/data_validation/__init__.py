"""Data validation utilities.

This package provides a lightweight stub that re-exports the
public validation helpers used throughout the project. The actual
implementations live in :mod:`ai_trading.data_validation.core`.
"""

from .core import (
    MarketDataValidator,
    ValidationSeverity,
    check_data_freshness,
    emergency_data_check,
    get_stale_symbols,
    is_valid_ohlcv,
    monitor_real_time_data_quality,
    should_halt_trading,
    validate_trade_log_integrity,
    validate_trading_data,
)

__all__ = [
    "MarketDataValidator",
    "ValidationSeverity",
    "check_data_freshness",
    "emergency_data_check",
    "get_stale_symbols",
    "is_valid_ohlcv",
    "monitor_real_time_data_quality",
    "should_halt_trading",
    "validate_trade_log_integrity",
    "validate_trading_data",
]
