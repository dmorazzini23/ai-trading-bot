from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

__all__ = [
    "check_data_freshness",
    "get_stale_symbols",
    "validate_trading_data",
    "emergency_data_check",
]


def check_data_freshness(
    df: pd.DataFrame, now: datetime | None = None, max_age_minutes: int = 30
) -> bool:
    """Return True if ``df`` has recent data."""
    if df is None or df.empty:
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    now = now or datetime.now(UTC)
    last = df.index.max().astimezone(UTC)
    age = (now - last).total_seconds() / 60.0
    return age <= max_age_minutes


def get_stale_symbols(
    frames: dict[str, pd.DataFrame],
    now: datetime | None = None,
    max_age_minutes: int = 30,
) -> list[str]:
    """Return symbols whose data is older than ``max_age_minutes``."""
    out: list[str] = []
    for sym, df in frames.items():
        if not check_data_freshness(df, now=now, max_age_minutes=max_age_minutes):
            out.append(sym)
    return out


def validate_trading_data(df: pd.DataFrame) -> bool:
    """Basic sanity checks for trading dataframes."""
    required = {"open", "high", "low", "close", "volume"}
    if df is None or df.empty:
        return False
    if not required.issubset(set(df.columns)):
        return False
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        return False
    return df.index.is_monotonic_increasing


def emergency_data_check(df: pd.DataFrame) -> bool:
    """Strict checks used for emergency trading safeguards."""
    if not validate_trading_data(df):
        return False
    if df.isna().any().any():
        return False
    return len(df) >= 5
