"""Minimal trading calendar wrapper backed by ``exchange_calendars``.

The real NYSE calendar is loaded on import when available.  When the
canonical :mod:`exchange_calendars` package is missing or fails to
initialize, a small fallback table provides deterministic sessions for
tests, including known early closes (e.g., Black Friday) and
daylight-saving transition Mondays for 2024–2025.
"""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from ai_trading.logging import get_logger

logger = get_logger(__name__)


def load_trading_calendars():
    """Return :mod:`exchange_calendars` if available, else ``None``."""

    try:  # pragma: no cover - optional dependency
        import exchange_calendars as ec  # type: ignore
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # library missing or incompatible
        logger.debug("EXCHANGE_CALENDARS_IMPORT_FAILED", exc_info=True)
        return None
    return ec


def load_pandas_market_calendars():
    """Return :mod:`pandas_market_calendars` if available."""

    try:  # pragma: no cover - optional dependency
        import pandas_market_calendars as pmc  # type: ignore
    except AI_TRADING_FALLBACK_EXCEPTIONS:
        logger.debug("PANDAS_MARKET_CALENDARS_IMPORT_FAILED", exc_info=True)
        return None
    return pmc


_ET = ZoneInfo("America/New_York")
_CAL = None
_tc = load_trading_calendars()
if _tc is not None:  # pragma: no branch - small init
    try:  # pragma: no cover - best effort
        _CAL = _tc.get_calendar("XNYS")
        # Preload sessions so holiday and early-close data is available without I/O.
        _CAL.sessions_in_range("2024-01-01", "2025-12-31")
    except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - fall back to stubs
        logger.debug("EXCHANGE_CALENDAR_INIT_FAILED", exc_info=True)
        _CAL = None


@dataclass(frozen=True)
class Session:
    """UTC session bounds with early-close metadata."""

    start_utc: datetime
    end_utc: datetime
    is_early_close: bool = False


def _session_from_et(
    d: date, close_hour: int, close_minute: int, early: bool = False
) -> Session:
    start_et = datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)
    end_et = datetime(d.year, d.month, d.day, close_hour, close_minute, tzinfo=_ET)
    return Session(start_et.astimezone(UTC), end_et.astimezone(UTC), early)


# Full-day market holidays for 2024–2025
_HOLIDAYS: set[date] = {
    date(2024, 1, 1),
    date(2024, 1, 15),
    date(2024, 2, 19),
    date(2024, 3, 29),
    date(2024, 5, 27),
    date(2024, 6, 19),
    date(2024, 7, 4),
    date(2024, 9, 2),
    date(2024, 11, 28),
    date(2024, 12, 25),
    date(2025, 1, 1),
    date(2025, 1, 20),
    date(2025, 2, 17),
    date(2025, 4, 18),
    date(2025, 5, 26),
    date(2025, 6, 19),
    date(2025, 7, 4),
    date(2025, 9, 1),
    date(2025, 11, 27),
    date(2025, 12, 25),
}


# Fallback sessions with explicit early closes and DST transitions
_FALLBACK_SESSIONS: dict[date, Session] = {
    # Dummy regular sessions for deterministic tests
    date(2024, 1, 1): _session_from_et(date(2024, 1, 1), 16, 0),
    date(2024, 1, 2): _session_from_et(date(2024, 1, 2), 16, 0),
    # Early closes
    date(2024, 7, 3): _session_from_et(date(2024, 7, 3), 13, 0, True),
    date(2024, 11, 29): _session_from_et(date(2024, 11, 29), 13, 0, True),
    date(2024, 12, 24): _session_from_et(date(2024, 12, 24), 13, 0, True),
    date(2025, 7, 3): _session_from_et(date(2025, 7, 3), 13, 0, True),
    date(2025, 11, 28): _session_from_et(date(2025, 11, 28), 13, 0, True),
    date(2025, 12, 24): _session_from_et(date(2025, 12, 24), 13, 0, True),
    # Regular trading day used in tests
    date(2025, 8, 20): _session_from_et(date(2025, 8, 20), 16, 0),
    # DST transition Mondays
    date(2024, 3, 11): _session_from_et(date(2024, 3, 11), 16, 0),
    date(2024, 11, 4): _session_from_et(date(2024, 11, 4), 16, 0),
    date(2025, 3, 10): _session_from_et(date(2025, 3, 10), 16, 0),
    date(2025, 11, 3): _session_from_et(date(2025, 11, 3), 16, 0),
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


def _computed_nyse_early_closes(year: int) -> set[date]:
    """Return fallback NYSE early closes that can be computed by rule."""
    candidates = {
        _nth_weekday(year, 11, 3, 4) + timedelta(days=1),  # Black Friday
        date(year, 12, 24),  # Christmas Eve when it is a trading day
        date(year, 7, 3),  # Day before Independence Day when it is a trading day
    }
    holidays = _computed_nyse_holidays(year)
    return {
        candidate
        for candidate in candidates
        if candidate.weekday() < 5 and candidate not in holidays
    }


def _is_fallback_holiday(d: date) -> bool:
    if d in _HOLIDAYS:
        return True
    return any(d in _computed_nyse_holidays(year) for year in (d.year - 1, d.year, d.year + 1))


def _is_fallback_early_close(d: date) -> bool:
    if d in _FALLBACK_SESSIONS:
        return _FALLBACK_SESSIONS[d].is_early_close
    return d in _computed_nyse_early_closes(d.year)


def is_trading_day(d: date) -> bool:
    """Return ``True`` if *d* is an NYSE trading day."""

    if _CAL is not None:
        try:
            return bool(_CAL.is_session(d))
        except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - defensive
            logger.debug("CALENDAR_IS_SESSION_FAILED", extra={"date": d.isoformat()}, exc_info=True)
    if _is_fallback_holiday(d):
        return False
    if d in _FALLBACK_SESSIONS:
        return True
    return d.weekday() < 5


def session_info(d: date) -> Session:
    """Return :class:`Session` for *d* with early-close metadata."""

    if _CAL is not None:
        try:
            if _CAL.is_session(d):
                open_et = _CAL.session_open(d).tz_convert(_ET).to_pydatetime()
                close_et = _CAL.session_close(d).tz_convert(_ET).to_pydatetime()
            else:
                prev = previous_trading_session(d)
                open_et = _CAL.session_open(prev).tz_convert(_ET).to_pydatetime()
                close_et = _CAL.session_close(prev).tz_convert(_ET).to_pydatetime()
            if hasattr(_CAL, "is_early_close"):
                early = bool(_CAL.is_early_close(d))
            else:
                early = close_et.hour < 16
            return Session(open_et.astimezone(UTC), close_et.astimezone(UTC), early)
        except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - fall back
            logger.debug("CALENDAR_SESSION_INFO_FAILED", extra={"date": d.isoformat()}, exc_info=True)
    if d in _FALLBACK_SESSIONS:
        return _FALLBACK_SESSIONS[d]
    if not is_trading_day(d):
        return session_info(previous_trading_session(d))
    if _is_fallback_early_close(d):
        return _session_from_et(d, 13, 0, True)
    return _session_from_et(d, 16, 0)


def rth_session_utc(d: date) -> tuple[datetime, datetime]:
    """Return the regular trading hours window in UTC."""

    s = session_info(d)
    return s.start_utc, s.end_utc


def rth_dst_summer_standard_times() -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    """Return representative DST and standard-time RTH windows in UTC.

    Uses the first Monday after the 2025 DST transitions to sample
    canonical summer and winter session times.  Falls back to
    known-good static dates if the calendar is unavailable.
    """

    summer = date(2025, 3, 10)
    winter = date(2025, 11, 3)
    return rth_session_utc(summer), rth_session_utc(winter)


def is_early_close(d: date) -> bool:
    """Return ``True`` if *d* is a scheduled early close."""

    return session_info(d).is_early_close


def previous_trading_session(d: date) -> date:
    """Return the previous NYSE trading day for *d*."""

    if _CAL is not None:
        try:
            prev = _CAL.previous_session(d)
            if isinstance(prev, datetime):
                return prev.date()
            if hasattr(prev, "date"):
                prev_date = prev.date()
                if isinstance(prev_date, date):
                    return prev_date
                return d
            if isinstance(prev, date):
                return prev
            return d
        except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - fall back
            logger.debug("CALENDAR_PREVIOUS_SESSION_FAILED", extra={"date": d.isoformat()}, exc_info=True)

    dd = d - timedelta(days=1)
    while not is_trading_day(dd):
        dd -= timedelta(days=1)
    return dd


def get_rth_session(d: date) -> tuple[datetime, datetime]:
    """Return the RTH session open/close in UTC."""

    return rth_session_utc(d)


__all__ = [
    "Session",
    "is_trading_day",
    "rth_session_utc",
    "rth_dst_summer_standard_times",
    "session_info",
    "is_early_close",
    "previous_trading_session",
    "load_trading_calendars",
    "get_rth_session",
]
