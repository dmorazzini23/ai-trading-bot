"""Lightweight data validation helpers used by critical tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trading.data_fetcher import get_bars

__all__ = [
    "check_data_freshness",
    "get_stale_symbols",
    "validate_trading_data",
    "emergency_data_check",
]


def check_data_freshness(
    df: pd.DataFrame,
    max_age_minutes: int = 15,
    now: datetime | None = None,
) -> tuple[bool, float]:
    """Return ``(is_fresh, age_min)`` based on latest timestamp."""
    if df is None or df.empty:
        return False, float("inf")
    now = now or datetime.now(UTC)
    ts = df["timestamp"].max() if "timestamp" in df.columns else df.index.max()
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age_min = (now - ts).total_seconds() / 60.0
    return age_min <= max_age_minutes, age_min


def get_stale_symbols(
    data_by_symbol: dict[str, pd.DataFrame],
    max_age_minutes: int = 15,
) -> list[str]:
    """Return list of symbols whose data is older than ``max_age_minutes``."""
    now = datetime.now(UTC)
    stale: list[str] = []
    for sym, df in data_by_symbol.items():
        fresh, _ = check_data_freshness(df, max_age_minutes, now)
        if not fresh:
            stale.append(sym)
    return stale


def validate_trading_data(
    data_by_symbol: dict[str, pd.DataFrame],
    min_rows: int = 10,
) -> bool:
    """Basic validation ensuring each dataframe has ``min_rows`` rows."""
    return all(df is not None and len(df) >= min_rows for df in data_by_symbol.values())


def emergency_data_check(
    symbols: list[str],
    min_bars: int = 10,
    fetcher: Callable[[str, datetime, datetime], pd.DataFrame] | None = None,
) -> dict[str, bool]:
    """Fetch a tiny window for ``symbols`` and report availability."""
    fetch = fetcher or get_bars
    end = datetime.now(UTC)
    start = end - timedelta(hours=1)
    result: dict[str, bool] = {}
    for sym in symbols:
        try:
            df = fetch(sym, start, end)
            result[sym] = bool(df is not None and len(df) >= min_bars)
        except Exception:
            result[sym] = False
    return result
