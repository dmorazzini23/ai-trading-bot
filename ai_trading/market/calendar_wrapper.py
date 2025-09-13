"""Minimal trading calendar wrapper backed by ``exchange_calendars``.

The real NYSE calendar is loaded on import when available.  When the
canonical :mod:`exchange_calendars` package is missing or fails to
initialize, a small fallback table provides deterministic sessions for
tests, including known early closes (e.g., Black Friday) and
daylight-saving transition Mondays for 2024–2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo


def load_trading_calendars():
    """Return :mod:`exchange_calendars` if available, else ``None``."""

    try:  # pragma: no cover - optional dependency
        import exchange_calendars as ec  # type: ignore
    except Exception:  # library missing or incompatible
        return None
    return ec


def load_pandas_market_calendars():
    """Return :mod:`pandas_market_calendars` if available."""

    try:  # pragma: no cover - optional dependency
        import pandas_market_calendars as pmc  # type: ignore
    except Exception:
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
    except Exception:  # pragma: no cover - fall back to stubs
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


def is_trading_day(d: date) -> bool:
    """Return ``True`` if *d* is an NYSE trading day."""

    if _CAL is not None:
        try:
            return bool(_CAL.is_session(d))
        except Exception:  # pragma: no cover - defensive
            pass
    if d in _HOLIDAYS:
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
        except Exception:  # pragma: no cover - fall back
            pass
    if d in _FALLBACK_SESSIONS:
        return _FALLBACK_SESSIONS[d]
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
            if hasattr(prev, "date"):
                return prev.date()
            return prev
        except Exception:  # pragma: no cover - fall back
            pass

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

