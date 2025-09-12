"""Backward-compatible wrapper for market calendar helpers.

This module simply re-exports utilities from :mod:`ai_trading.utils.market_calendar`.
"""

from ai_trading.utils.market_calendar import (
    Session,
    is_trading_day,
    rth_session_utc,
    session_info,
    is_early_close,
    previous_trading_session,
)

__all__ = [
    "Session",
    "is_trading_day",
    "rth_session_utc",
    "session_info",
    "is_early_close",
    "previous_trading_session",
]
