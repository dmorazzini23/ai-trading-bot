"""Canonical fixed-interval, exchange-session label eligibility."""
import pandas as pd

def valid_five_minute_labels(index: pd.DatetimeIndex, horizon_bars: int) -> pd.Series:
    """Labels require consecutive five-minute bars within one local session date."""
    if (isinstance(horizon_bars, bool) or not isinstance(horizon_bars, int) or horizon_bars < 1
            or index.tz is None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing):
        raise ValueError('Invalid label horizon or timestamp history')
    stamps = pd.Series(index, index=index)
    step = stamps.shift(-1).sub(stamps).eq(pd.Timedelta(minutes=5))
    valid = pd.Series(True, index=index)
    for offset in range(horizon_bars):
        valid &= step.shift(-offset, fill_value=False)
    dates = pd.Series(index.tz_convert('America/New_York').date, index=index)
    from ai_trading.utils.market_calendar import is_trading_day, session_info
    regular = pd.Series(False, index=index)
    for day in set(dates):
        if is_trading_day(day):
            session = session_info(day)
            regular |= (index >= session.start_utc) & (index < session.end_utc)
    regular &= (index.minute % 5 == 0) & (index.second == 0) & (index.microsecond == 0) & (index.nanosecond == 0)
    return valid & dates.eq(dates.shift(-horizon_bars)) & regular & regular.shift(-horizon_bars, fill_value=False)
