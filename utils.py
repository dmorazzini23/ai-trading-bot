"""Utility functions for common operations across the bot."""

import warnings
import os

import pandas as pd
from datetime import datetime, time
try:
    from tzlocal import get_localzone
except ImportError:  # pragma: no cover - optional dependency
    import logging, pytz
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
    """Return True if within weekday US market hours (9:30–16:00 ET)."""  # FIXED
    now_et = (now or datetime.now(tz=EASTERN_TZ)).astimezone(EASTERN_TZ)
    if now_et.weekday() >= 5:
        return False
    current = now_et.time()
    return MARKET_OPEN_TIME <= current <= MARKET_CLOSE_TIME


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def data_filepath(filename: str) -> str:
    return os.path.join(BASE_PATH, 'data', filename)


def convert_to_local(df: pd.DataFrame) -> pd.DataFrame:
    local_tz = get_localzone()
    return df.tz_convert(local_tz)
