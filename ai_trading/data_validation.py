"""Utility functions for validating market data."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pandas as pd

__all__ = [
    "check_data_freshness",
    "validate_trading_data",
    "get_stale_symbols",
    "emergency_data_check",
]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def check_data_freshness(
    df: pd.DataFrame,
    freshness_minutes: int = 5,
    now: datetime | None = None,
) -> bool:
    """Return ``True`` if ``df`` has data within ``freshness_minutes``."""
    now = now or _utcnow()
    if df is None or df.empty:
        return False
    ts = df.index.max()
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age = now - ts
    return age <= timedelta(minutes=freshness_minutes)


def get_stale_symbols(
    last_ts: Mapping[str, datetime] | pd.DataFrame,
    now: datetime | None = None,
    threshold_minutes: int = 5,
) -> list[str]:
    """Return symbols whose timestamps exceed ``threshold_minutes`` age."""
    now = now or _utcnow()
    stale: list[str] = []
    if isinstance(last_ts, Mapping):
        for sym, ts in last_ts.items():
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if now - ts > timedelta(minutes=threshold_minutes):
                stale.append(sym)
        return stale
    ts_col = "timestamp" if "timestamp" in last_ts.columns else last_ts.index.name
    for sym, grp in last_ts.groupby("symbol"):
        ts = grp[ts_col].max() if ts_col in grp else grp.index.max()
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if now - ts > timedelta(minutes=threshold_minutes):
            stale.append(sym)
    return stale


def emergency_data_check(df: pd.DataFrame, strict: bool = False) -> bool:
    """Return ``True`` if ``df`` passes emergency validation checks."""
    if df is None or df.empty:
        return False
    if strict:
        return df.dropna().shape[0] > 0
    return True


def validate_trading_data(
    df: pd.DataFrame,
    *,
    min_rows: int = 1,
    allow_na: bool = True,
) -> bool:
    """Basic sanity checks for OHLCV dataframes."""
    if df is None or df.shape[0] < min_rows:
        return False
    if not allow_na and df.isna().any().any():
        return False
    return True
