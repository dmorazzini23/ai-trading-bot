"""Utility functions for common operations across the bot."""

import warnings
import os
import re

import pandas as pd
from datetime import datetime, date, time, timezone

try:
    from tzlocal import get_localzone
except ImportError:  # pragma: no cover - optional dependency
    import logging
    import pytz

    logging.warning("tzlocal not installed; defaulting to UTC")

    def get_localzone():
        return pytz.UTC


from zoneinfo import ZoneInfo
import threading

warnings.filterwarnings("ignore", category=FutureWarning)

MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(16, 0)
EASTERN_TZ = ZoneInfo("America/New_York")

# Lock protecting portfolio state across threads
portfolio_lock = threading.Lock()
# Lock protecting model updates
model_lock = threading.Lock()


def get_latest_close(df: pd.DataFrame) -> float:
    """Return last closing price or 1.0 if unavailable."""
    if df is None or df.empty:
        return 1.0
    if "close" in df.columns:
        last = df["close"].iloc[-1]
    else:
        return 1.0
    if pd.isna(last) or last == 0:
        return 1.0
    return float(last)


def is_market_open(now: datetime | None = None) -> bool:
    """Return True if current time is within NYSE trading hours."""
    try:
        import pandas_market_calendars as mcal

        check_time = (now or datetime.now(tz=EASTERN_TZ)).astimezone(EASTERN_TZ)
        cal = getattr(mcal, "get_calendar", None)
        if cal is None:
            raise AttributeError
        cal = cal("NYSE")
        sched = cal.schedule(
            start_date=check_time.date(), end_date=check_time.date()
        )
        if sched.empty:
            return False  # holiday or weekend
        market_open = sched.iloc[0]["market_open"].tz_convert(EASTERN_TZ).time()
        market_close = sched.iloc[0]["market_close"].tz_convert(EASTERN_TZ).time()
        current = check_time.time()
        return market_open <= current <= market_close
    except Exception:
        # Fallback to simple weekday/time check when calendar unavailable
        now_et = (now or datetime.now(tz=EASTERN_TZ)).astimezone(EASTERN_TZ)
        if now_et.weekday() >= 5:
            return False
        current = now_et.time()
        return MARKET_OPEN_TIME <= current <= MARKET_CLOSE_TIME


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def data_filepath(filename: str) -> str:
    return os.path.join(BASE_PATH, "data", filename)


def convert_to_local(df: pd.DataFrame) -> pd.DataFrame:
    local_tz = get_localzone()
    assert df.index.tz is not None, "DataFrame index must be timezone aware"
    return df.tz_convert(local_tz)


def ensure_utc(dt: datetime | date) -> datetime:
    """Return a timezone-aware UTC datetime for ``dt``."""
    assert isinstance(dt, (datetime, date)), "dt must be date or datetime"
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(dt, date):
        return datetime.combine(dt, time.min, tzinfo=timezone.utc)
    raise TypeError(f"Unsupported type for ensure_utc: {type(dt)!r}")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def safe_to_datetime(values) -> pd.DatetimeIndex | None:
    """Return ``DatetimeIndex`` if values are parseable, else ``None``."""
    if values is None or len(values) == 0:
        return None
    sample = values[0]
    if isinstance(sample, tuple):
        sample = next((v for v in sample if isinstance(v, str) or hasattr(v, "year")), sample)
    if isinstance(sample, str) and not _DATE_RE.match(sample):
        return None
    try:
        idx = pd.to_datetime(values, errors="coerce", utc=True)
    except Exception:
        return None
    idx = idx.tz_convert(None)
    if idx.isnull().all():
        return None
    return idx
