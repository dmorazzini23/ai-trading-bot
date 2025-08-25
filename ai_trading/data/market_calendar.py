from __future__ import annotations
from dataclasses import dataclass
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo
from ai_trading.utils.lazy_imports import load_pandas, load_pandas_market_calendars

_ET = ZoneInfo('America/New_York')
_CAL = None

def _get_calendar():
    """Load and cache the trading calendar on demand."""
    global _CAL
    if _CAL is None:
        mcal = load_pandas_market_calendars()
        if mcal is None:
            return None
        _CAL = mcal.get_calendar('XNYS')
    return _CAL

@dataclass(frozen=True)
class Session:
    start_utc: datetime
    end_utc: datetime

def is_trading_day(d: date) -> bool:
    """Return True if *d* is a trading day."""
    cal = _get_calendar()
    if cal is not None:
        days = cal.valid_days(start_date=d, end_date=d)
        return len(days) == 1
    return d.weekday() < 5

def _pmc_session_utc(d: date) -> Session:
    """Fetch accurate session times via pandas-market-calendars."""
    cal = _get_calendar()
    if cal is None:
        raise RuntimeError("pandas_market_calendars not available")
    pd = load_pandas()
    sched = cal.schedule(start_date=d, end_date=d, tz=_ET)
    if sched.empty:
        prev = previous_trading_session(d)
        sched = cal.schedule(start_date=prev, end_date=prev, tz=_ET)
    open_et = sched.iloc[0]['market_open'].to_pydatetime().astimezone(_ET)
    close_et = sched.iloc[0]['market_close'].to_pydatetime().astimezone(_ET)
    return Session(open_et.astimezone(UTC), close_et.astimezone(UTC))

def rth_session_utc(d: date) -> tuple[datetime, datetime]:
    """Return the Regular Trading Hours window in UTC."""
    cal = _get_calendar()
    if cal is not None:
        s = _pmc_session_utc(d)
        return (s.start_utc, s.end_utc)
    start_et = datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)
    end_et = datetime(d.year, d.month, d.day, 16, 0, tzinfo=_ET)
    return (start_et.astimezone(UTC), end_et.astimezone(UTC))

def previous_trading_session(d: date) -> date:
    """Return the previous trading day for *d*."""
    cal = _get_calendar()
    if cal is not None:
        days = cal.valid_days(start_date=d.replace(day=1), end_date=d)
        if len(days) == 0:
            from datetime import timedelta
            back = d.replace(day=1) - timedelta(days=1)
            days = cal.valid_days(start_date=back.replace(day=1), end_date=back)
        return days[-1].date()
    from datetime import timedelta
    dd = d
    while True:
        dd = dd - timedelta(days=1)
        if dd.weekday() < 5:
            return dd
