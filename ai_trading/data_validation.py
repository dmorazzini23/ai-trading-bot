"""Lightweight data validation helpers used by tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trading.data_fetcher import get_bars

__all__ = [
    "check_data_freshness",
    "validate_trading_data",
    "get_stale_symbols",
    "emergency_data_check",
]


def check_data_freshness(
    df: pd.DataFrame,
    *,
    now: datetime | None = None,
    max_age_minutes: int = 5,
) -> bool:
    """Return ``True`` if ``df`` has data within ``max_age_minutes``."""
    if df is None or df.empty:
        return False
    ts = df["timestamp"].max() if "timestamp" in df.columns else df.index.max()
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    now_dt = now or datetime.now(UTC)
    return now_dt - ts <= timedelta(minutes=max_age_minutes)


def validate_trading_data(
    df: pd.DataFrame,
    *,
    required_cols: Iterable[str] = ("open", "high", "low", "close", "volume"),
) -> tuple[bool, list[str]]:
    """Check that ``df`` has ``required_cols`` and is not empty."""
    issues: list[str] = []
    if df is None or df.empty:
        issues.append("empty")
        return False, issues
    missing = [c for c in required_cols if c not in df.columns]
    issues.extend(missing)
    return (not issues, issues)


def get_stale_symbols(
    frames: Mapping[str, pd.DataFrame],
    *,
    max_age_minutes: int = 5,
) -> list[str]:
    """Return symbols whose data is older than ``max_age_minutes``."""
    now = datetime.now(UTC)
    return [
        sym
        for sym, df in frames.items()
        if not check_data_freshness(df, now=now, max_age_minutes=max_age_minutes)
    ]


def emergency_data_check(
    symbols: Sequence[str],
    *,
    fetcher: Callable[..., pd.DataFrame] | None = None,
) -> bool:
    """Fetch a tiny window for ``symbols`` and ensure data exists."""
    fetch = fetcher or get_bars
    end = datetime.now(UTC)
    start = end - timedelta(minutes=1)
    for sym in symbols:
        try:
            df = fetch(sym, start, end)
            if df is None or df.empty:
                return False
        except Exception:
            return False
    return True
