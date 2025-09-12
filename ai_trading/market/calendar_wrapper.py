"""Backward-compatible wrapper for market calendar helpers.

Provides minimal functionality when :mod:`pandas_market_calendars` is missing by
falling back to an internal session table used in tests.
"""

from ai_trading.utils.lazy_imports import load_pandas_market_calendars as _load_pmc
from ai_trading.utils.market_calendar import (
    Session,
    is_trading_day,
    rth_session_utc,
    session_info,
    is_early_close,
    previous_trading_session,
)


def load_pandas_market_calendars():
    """Return :mod:`pandas_market_calendars` or a lightweight stub."""
    pmc = _load_pmc()
    if pmc is not None:
        return pmc

    class _Stub:
        def get_calendar(self, name):  # pragma: no cover - simple stub
            return self

        def schedule(self, start_date, end_date):  # pragma: no cover - simple stub
            return {}

    return _Stub()


def get_rth_session(d):
    """Return the RTH session open/close in UTC."""
    return rth_session_utc(d)


__all__ = [
    "Session",
    "is_trading_day",
    "rth_session_utc",
    "session_info",
    "is_early_close",
    "previous_trading_session",
    "load_pandas_market_calendars",
    "get_rth_session",
]
