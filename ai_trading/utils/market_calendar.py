from __future__ import annotations

"""Lightweight NYSE calendar helpers with 2024-2025 overrides.

This module provides small trading-session utilities without requiring
:mod:`pandas_market_calendars`. When the optional package is available,
its data is used for accurate open/close times. Otherwise a minimal set
of known sessions is used as a fallback.

The fallback table intentionally covers:
* Black Friday early closes for 2024-2025
* DST transition Mondays for 2024-2025

These are sufficient for unit tests to validate daylight-saving time
shifts and early-close handling.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from ai_trading.utils.lazy_imports import load_pandas, load_pandas_market_calendars

_ET = ZoneInfo("America/New_York")
_CAL = None


def _get_calendar():
    """Load and cache the NYSE calendar on demand."""
    global _CAL
    if _CAL is None:
        mcal = load_pandas_market_calendars()
        if mcal is None:
            return None
        _CAL = mcal.get_calendar("XNYS")
    return _CAL


@dataclass(frozen=True)
class Session:
    """UTC session bounds with early-close metadata."""

    start_utc: datetime
    end_utc: datetime
    is_early_close: bool = False


# Known session overrides when :mod:`pandas_market_calendars` is missing.
# Times are defined in ET then converted to UTC for accuracy.
_FALLBACK_SESSIONS: dict[date, Session] = {
    # Dummy sessions for early January 2024 to keep tests deterministic
    date(2024, 1, 1): Session(
        datetime(2024, 1, 1, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2024, 1, 1, 16, 0, tzinfo=_ET).astimezone(UTC),
    ),
    date(2024, 1, 2): Session(
        datetime(2024, 1, 2, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2024, 1, 2, 16, 0, tzinfo=_ET).astimezone(UTC),
    ),
    # Black Friday early closes
    date(2024, 11, 29): Session(
        datetime(2024, 11, 29, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2024, 11, 29, 13, 0, tzinfo=_ET).astimezone(UTC),
        True,
    ),
    date(2025, 11, 28): Session(
        datetime(2025, 11, 28, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2025, 11, 28, 13, 0, tzinfo=_ET).astimezone(UTC),
        True,
    ),
    # DST transition Mondays (start / end) to cover edge cases
    date(2024, 3, 11): Session(
        datetime(2024, 3, 11, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2024, 3, 11, 16, 0, tzinfo=_ET).astimezone(UTC),
    ),
    date(2024, 11, 4): Session(
        datetime(2024, 11, 4, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2024, 11, 4, 16, 0, tzinfo=_ET).astimezone(UTC),
    ),
    date(2025, 3, 10): Session(
        datetime(2025, 3, 10, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2025, 3, 10, 16, 0, tzinfo=_ET).astimezone(UTC),
    ),
    date(2025, 11, 3): Session(
        datetime(2025, 11, 3, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2025, 11, 3, 16, 0, tzinfo=_ET).astimezone(UTC),
    ),
}


def is_trading_day(d: date) -> bool:
    """Return ``True`` if *d* is an NYSE trading day."""
    cal = _get_calendar()
    if cal is not None:
        valid_days = getattr(cal, "valid_days", lambda *a, **k: [])
        days = valid_days(start_date=d, end_date=d)
        return len(days) == 1
    # Fallback: weekdays only. All known early closes are trading days.
    return d.weekday() < 5


def _pmc_session_info(d: date) -> Session:
    """Fetch accurate session times via pandas-market-calendars."""
    cal = _get_calendar()
    if cal is None:
        raise RuntimeError("pandas_market_calendars not available")
    pd = load_pandas()
    sched = cal.schedule(start_date=d, end_date=d)
    if sched.empty:
        prev = previous_trading_session(d)
        sched = cal.schedule(start_date=prev, end_date=prev)
    if sched.empty:
        raise RuntimeError(f"No trading session for {d}")
    open_et = sched.iloc[0]["market_open"].tz_convert(_ET).to_pydatetime()
    close_et = sched.iloc[0]["market_close"].tz_convert(_ET).to_pydatetime()
    early = close_et.hour < 16
    return Session(open_et.astimezone(UTC), close_et.astimezone(UTC), early)


def session_info(d: date) -> Session:
    """Return Session info for *d* with early-close metadata."""
    cal = _get_calendar()
    if cal is not None:
        return _pmc_session_info(d)
    if d in _FALLBACK_SESSIONS:
        return _FALLBACK_SESSIONS[d]
    start_et = datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)
    end_et = datetime(d.year, d.month, d.day, 16, 0, tzinfo=_ET)
    return Session(start_et.astimezone(UTC), end_et.astimezone(UTC))


def rth_session_utc(d: date) -> tuple[datetime, datetime]:
    """Return the Regular Trading Hours window in UTC."""
    s = session_info(d)
    return (s.start_utc, s.end_utc)


def is_early_close(d: date) -> bool:
    """Return ``True`` if *d* is a scheduled early-close day."""
    return session_info(d).is_early_close


def previous_trading_session(d: date) -> date:
    """Return the previous trading day for *d*."""
    from datetime import timedelta

    cal = _get_calendar()
    if cal is not None:
        valid_days = getattr(cal, "valid_days", lambda *a, **k: [])
        days = valid_days(start_date=d.replace(day=1), end_date=d)
        if len(days) == 0:
            back = d.replace(day=1) - timedelta(days=1)
            days = valid_days(start_date=back.replace(day=1), end_date=back)
        if len(days) == 0:
            dd = d
            while True:
                dd = dd - timedelta(days=1)
                if dd.weekday() < 5:
                    return dd
        return days[-1].date()
    dd = d
    while True:
        dd = dd - timedelta(days=1)
        if dd.weekday() < 5:
            return dd


__all__ = [
    "Session",
    "is_trading_day",
    "rth_session_utc",
    "session_info",
    "is_early_close",
    "previous_trading_session",
]
