from __future__ import annotations

# NYSE trading calendar helper with DST/holiday awareness.  # AI-AGENT-REF: calendar wrapper
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

_HAVE_PMC = False
try:  # ImportError-only guard
    import pandas as pd  # type: ignore
    import pandas_market_calendars as mcal  # type: ignore

    _CAL = mcal.get_calendar("XNYS")
    _HAVE_PMC = True
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
    _CAL = None  # type: ignore


@dataclass(frozen=True)
class Session:
    start_utc: datetime
    end_utc: datetime


def is_trading_day(d: date) -> bool:
    """Return True if *d* is a trading day."""  # AI-AGENT-REF: trading day helper

    if _HAVE_PMC:
        days = _CAL.valid_days(start_date=d, end_date=d)
        return len(days) == 1
    # Fallback: Mon–Fri
    return d.weekday() < 5


def _pmc_session_utc(d: date) -> Session:
    """Fetch accurate session times via pandas-market-calendars."""

    # Build a one-day schedule; respects holidays and early closes
    sched = _CAL.schedule(start_date=d, end_date=d, tz=_ET)
    if sched.empty:
        # Not a trading day — choose previous valid and return its session
        prev = previous_trading_session(d)
        sched = _CAL.schedule(start_date=prev, end_date=prev, tz=_ET)
    open_et = sched.iloc[0]["market_open"].to_pydatetime().astimezone(_ET)
    close_et = sched.iloc[0]["market_close"].to_pydatetime().astimezone(_ET)
    return Session(open_et.astimezone(UTC), close_et.astimezone(UTC))


def rth_session_utc(d: date) -> tuple[datetime, datetime]:
    """Return the Regular Trading Hours window in UTC."""  # AI-AGENT-REF

    if _HAVE_PMC:  # accurate path
        s = _pmc_session_utc(d)
        return s.start_utc, s.end_utc
    # fallback path (no holidays/early-close knowledge)
    start_et = datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)
    end_et = datetime(d.year, d.month, d.day, 16, 0, tzinfo=_ET)
    return start_et.astimezone(UTC), end_et.astimezone(UTC)


def previous_trading_session(d: date) -> date:
    """Return the previous trading day for *d*."""  # AI-AGENT-REF

    if _HAVE_PMC:
        days = _CAL.valid_days(start_date=d.replace(day=1), end_date=d)
        if len(days) == 0:
            from datetime import timedelta

            back = d.replace(day=1) - timedelta(days=1)
            days = _CAL.valid_days(start_date=back.replace(day=1), end_date=back)
        return days[-1].date()  # type: ignore[index]
    # Fallback: step back to previous weekday
    from datetime import timedelta

    dd = d
    while True:
        dd = dd - timedelta(days=1)
        if dd.weekday() < 5:
            return dd

