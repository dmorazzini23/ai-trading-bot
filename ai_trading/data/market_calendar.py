from __future__ import annotations
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo
_ET = ZoneInfo('America/New_York')
_HAVE_PMC = False
try:
    import pandas as pd
    import pandas_market_calendars as mcal
    _CAL = mcal.get_calendar('XNYS')
    _HAVE_PMC = True
except ImportError:
    pd = None
    _CAL = None

@dataclass(frozen=True)
class Session:
    start_utc: datetime
    end_utc: datetime

def is_trading_day(d: date) -> bool:
    """Return True if *d* is a trading day."""
    if _HAVE_PMC:
        days = _CAL.valid_days(start_date=d, end_date=d)
        return len(days) == 1
    return d.weekday() < 5

def _pmc_session_utc(d: date) -> Session:
    """Fetch accurate session times via pandas-market-calendars."""
    sched = _CAL.schedule(start_date=d, end_date=d, tz=_ET)
    if sched.empty:
        prev = previous_trading_session(d)
        sched = _CAL.schedule(start_date=prev, end_date=prev, tz=_ET)
    open_et = sched.iloc[0]['market_open'].to_pydatetime().astimezone(_ET)
    close_et = sched.iloc[0]['market_close'].to_pydatetime().astimezone(_ET)
    return Session(open_et.astimezone(UTC), close_et.astimezone(UTC))

def rth_session_utc(d: date) -> tuple[datetime, datetime]:
    """Return the Regular Trading Hours window in UTC."""
    if _HAVE_PMC:
        s = _pmc_session_utc(d)
        return (s.start_utc, s.end_utc)
    start_et = datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)
    end_et = datetime(d.year, d.month, d.day, 16, 0, tzinfo=_ET)
    return (start_et.astimezone(UTC), end_et.astimezone(UTC))

def previous_trading_session(d: date) -> date:
    """Return the previous trading day for *d*."""
    if _HAVE_PMC:
        days = _CAL.valid_days(start_date=d.replace(day=1), end_date=d)
        if len(days) == 0:
            from datetime import timedelta
            back = d.replace(day=1) - timedelta(days=1)
            days = _CAL.valid_days(start_date=back.replace(day=1), end_date=back)
        return days[-1].date()
    from datetime import timedelta
    dd = d
    while True:
        dd = dd - timedelta(days=1)
        if dd.weekday() < 5:
            return dd