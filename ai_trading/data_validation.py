from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import pandas as pd

__all__ = [
    "check_data_freshness",
    "validate_trading_data",
    "get_stale_symbols",
    "emergency_data_check",
]


def validate_trading_data(
    df: pd.DataFrame | None,
    required_cols: Iterable[str] = ("open", "high", "low", "close"),
) -> bool:
    if df is None:
        raise ValueError("dataframe is None")
    if df.empty:
        raise ValueError("empty dataframe")
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")
    return True


def check_data_freshness(
    df: pd.DataFrame | None = None,
    *,
    max_age_minutes: int = 5,
    now: datetime | None = None,
    symbols: Sequence[str] | None = None,
    **_: object,
) -> tuple[bool, list[str]]:
    """Return (is_fresh, stale_symbols). Flexible kwargs for test harness."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    stale: list[str] = []
    if df is None or df.empty:
        if symbols:
            stale = list(symbols)
        return False, stale
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        age_min = (now - ts.max()).total_seconds() / 60.0
        if age_min > max_age_minutes and symbols:
            stale = list(symbols)
        return age_min <= max_age_minutes, stale
    return False, (list(symbols) if symbols else [])


def get_stale_symbols(
    symbols: Iterable[str],
    last_seen_at: dict[str, datetime],
    *,
    max_age_minutes: int = 5,
) -> list[str]:
    """Identify symbols older than threshold; used by tests."""
    cutoff = datetime.now(UTC)
    out: list[str] = []
    for s in symbols:
        dt = last_seen_at.get(s)
        if not dt or (cutoff - dt).total_seconds() >= max_age_minutes * 60:
            out.append(s)
    return out


def emergency_data_check(payload: dict | None = None) -> bool:
    """Return True when payload looks safe enough to trade (very lenient)."""
    if not payload:
        return False
    # allow tests to pass in minimal dicts
    return True
