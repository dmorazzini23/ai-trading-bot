"""Utility functions for common operations across the bot."""

import pandas as pd
from datetime import datetime, time
from zoneinfo import ZoneInfo


def get_latest_close(df: pd.DataFrame) -> float:
    """Return last closing price or 1.0 if unavailable."""
    if df is None or df.empty:
        return 1.0
    if "close" in df.columns:
        last = df["close"].iloc[-1]
    elif "Close" in df.columns:
        last = df["Close"].iloc[-1]
    else:
        return 1.0
    if pd.isna(last) or last == 0:
        return 1.0
    return float(last)


def is_market_open() -> bool:
    """Return True if current time in New York is between 9:30 and 16:00."""
    now = datetime.now(ZoneInfo("America/New_York"))
    start = time(9, 30)
    end = time(16, 0)
    return start <= now.time() <= end
