"""Thin wrapper around :mod:`ai_trading.utils.market_calendar`.

The data module historically provided market-calendar helpers.  It now
re-exports the shared utilities so existing imports continue to work.
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
