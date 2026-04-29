from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

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
from datetime import UTC, date, datetime, timedelta
from typing import Any
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
    # Final trading days for 2023 used by previous-session fallbacks
    date(2023, 12, 28): Session(
        datetime(2023, 12, 28, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2023, 12, 28, 16, 0, tzinfo=_ET).astimezone(UTC),
    ),
    date(2023, 12, 29): Session(
        datetime(2023, 12, 29, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2023, 12, 29, 16, 0, tzinfo=_ET).astimezone(UTC),
    ),
    # Regular session for early January 2024 to keep tests deterministic
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
    # Regular trading day used in tests
    date(2025, 8, 20): Session(
        datetime(2025, 8, 20, 9, 30, tzinfo=_ET).astimezone(UTC),
        datetime(2025, 8, 20, 16, 0, tzinfo=_ET).astimezone(UTC),
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


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    actual = date(year, month, day)
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _easter_date(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _computed_nyse_holidays(year: int) -> set[date]:
    return {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_date(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed_holiday(year, 6, 19),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed_holiday(year, 12, 25),
    }


def _is_fallback_holiday(d: date) -> bool:
    return any(d in _computed_nyse_holidays(year) for year in (d.year - 1, d.year, d.year + 1))


def is_trading_day(d: date) -> bool:
    """Return ``True`` if *d* is an NYSE trading day."""
    cal = _get_calendar()
    if cal is not None:
        valid_days = getattr(cal, "valid_days", None)
        if callable(valid_days):
            try:
                days = valid_days(start_date=d, end_date=d)
            except AI_TRADING_FALLBACK_EXCEPTIONS:
                pass
            else:
                try:
                    return len(days) == 1
                except TypeError:
                    try:
                        materialized = list(days)
                    except TypeError:
                        pass
                    else:
                        return len(materialized) == 1
        return d.weekday() < 5
    if _is_fallback_holiday(d):
        return False
    return d.weekday() < 5


def _pmc_session_info(d: date, *, allow_previous_session: bool = False) -> Session:
    """Fetch accurate session times via pandas-market-calendars."""
    cal = _get_calendar()
    if cal is None:
        raise RuntimeError("pandas_market_calendars not available")
    sched = cal.schedule(start_date=d, end_date=d)
    if sched.empty:
        if not allow_previous_session:
            raise ValueError(f"not_trading_session: {d.isoformat()}")
        prev = previous_trading_session(d)
        sched = cal.schedule(start_date=prev, end_date=prev)
    if sched.empty:
        raise ValueError(f"not_trading_session: {d.isoformat()}")
    open_et = sched.iloc[0]["market_open"].tz_convert(_ET).to_pydatetime()
    close_et = sched.iloc[0]["market_close"].tz_convert(_ET).to_pydatetime()
    early = close_et.hour < 16
    return Session(open_et.astimezone(UTC), close_et.astimezone(UTC), early)


def session_info(d: date, *, allow_previous_session: bool = False) -> Session:
    """Return Session info for *d* with early-close metadata."""
    cal = _get_calendar()
    if cal is not None:
        return _pmc_session_info(d, allow_previous_session=allow_previous_session)
    if not is_trading_day(d):
        if allow_previous_session:
            return session_info(previous_trading_session(d))
        raise ValueError(f"not_trading_session: {d.isoformat()}")
    if d in _FALLBACK_SESSIONS:
        return _FALLBACK_SESSIONS[d]
    start_et = datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)
    end_et = datetime(d.year, d.month, d.day, 16, 0, tzinfo=_ET)
    return Session(start_et.astimezone(UTC), end_et.astimezone(UTC))


def rth_session_utc(
    d: date,
    *,
    allow_previous_session: bool = False,
) -> tuple[datetime, datetime]:
    """Return the Regular Trading Hours window in UTC."""
    s = session_info(d, allow_previous_session=allow_previous_session)
    return (s.start_utc, s.end_utc)


def is_early_close(d: date) -> bool:
    """Return ``True`` if *d* is a scheduled early-close day."""
    return session_info(d).is_early_close


def previous_trading_session(d: date) -> date:
    """Return the previous trading day for *d*."""
    cal = _get_calendar()
    end_date = d - timedelta(days=1)
    if cal is not None:
        valid_days = getattr(cal, "valid_days", None)
        days: list[Any] = []
        if callable(valid_days):
            days = list(valid_days(start_date=end_date.replace(day=1), end_date=end_date))
        if len(days) == 0:
            back = end_date.replace(day=1) - timedelta(days=1)
            if callable(valid_days):
                days = list(valid_days(start_date=back.replace(day=1), end_date=back))
        if len(days) == 0:
            dd = d
            while True:
                dd = dd - timedelta(days=1)
                if dd.weekday() < 5:
                    return dd
        last_day = days[-1]
        if isinstance(last_day, datetime):
            candidate = last_day.date()
            return candidate if candidate < d else previous_trading_session(candidate)
        if hasattr(last_day, "date"):
            candidate_date = last_day.date()
            if isinstance(candidate_date, date):
                return candidate_date if candidate_date < d else previous_trading_session(candidate_date)
        if isinstance(last_day, date):
            return last_day if last_day < d else previous_trading_session(last_day)
        return end_date
    dd = d
    while True:
        dd = dd - timedelta(days=1)
        if is_trading_day(dd):
            return dd


__all__ = [
    "Session",
    "is_trading_day",
    "rth_session_utc",
    "session_info",
    "is_early_close",
    "previous_trading_session",
]
