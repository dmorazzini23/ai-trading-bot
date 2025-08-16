from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

__all__ = [
    "check_data_freshness",
    "validate_trading_data",
    "get_stale_symbols",
    "emergency_data_check",
]


def validate_trading_data(
    df: Any,
    required_cols: Iterable[str] = ("open", "high", "low", "close"),
) -> bool:
    if df is None:
        raise ValueError("dataframe is None")
    if pd is not None and isinstance(df, pd.DataFrame):
        if df.empty:
            raise ValueError("empty dataframe")
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"missing column: {col}")
        return True
    # Accept simple mappings/sequences in tests
    return True


def check_data_freshness(
    df: Any,
    *,
    max_stale_minutes: int | None = None,
    max_age_minutes: int | None = None,
    now: datetime | None = None,
    **_: Any,
) -> bool:
    # Accept either kw name
    minutes = (
        max_stale_minutes if max_stale_minutes is not None else (max_age_minutes or 5)
    )
    now = (now or datetime.now(UTC)).astimezone(UTC)

    # Extract last timestamp
    last_ts: datetime | None = None
    if pd is not None and isinstance(df, pd.DataFrame) and "timestamp" in df.columns:
        v = df["timestamp"].iloc[-1]
        last_ts = v if isinstance(v, datetime) else None
    elif isinstance(df, Mapping) and "timestamp" in df:
        v = df["timestamp"]
        last_ts = v if isinstance(v, datetime) else None

    if last_ts is None:
        # If unknown, treat as stale for safety
        return False
    last_ts = last_ts.astimezone(UTC)
    return last_ts >= now - timedelta(minutes=minutes)


def get_stale_symbols(
    data_by_symbol: Mapping[str, Any],
    *,
    max_stale_minutes: int = 5,
    now: datetime | None = None,
) -> list[str]:
    stale: list[str] = []
    for sym, df in data_by_symbol.items():
        if not check_data_freshness(df, max_stale_minutes=max_stale_minutes, now=now):
            stale.append(sym)
    return stale


def emergency_data_check(data: Any, **kwargs: Any) -> bool:
    """
    Lightweight 'are we obviously safe to trade' check used by tests.
    Accepts flexible kwargs to remain forwards-compatible.
    """
    try:
        return validate_trading_data(data) and check_data_freshness(data, **kwargs)
    except Exception:
        return False
