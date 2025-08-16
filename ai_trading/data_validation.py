"""Lightweight data validation utilities for tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# AI-AGENT-REF: fresh data check


def check_data_freshness(last_updated: datetime, *, max_age_minutes: int = 15) -> bool:
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=UTC)
    return (datetime.now(UTC) - last_updated) <= timedelta(minutes=max_age_minutes)


# AI-AGENT-REF: stale symbol helper


def get_stale_symbols(
    symbol_updates: dict[str, datetime], *, max_age_minutes: int = 15
) -> list[str]:
    return [
        s
        for s, ts in symbol_updates.items()
        if not check_data_freshness(ts, max_age_minutes=max_age_minutes)
    ]


# AI-AGENT-REF: basic structural validation


def validate_trading_data(df) -> bool:
    required = {"symbol", "timestamp"}
    return hasattr(df, "columns") and required.issubset(set(df.columns))


# AI-AGENT-REF: critical data check


def emergency_data_check(df) -> bool:
    if not validate_trading_data(df) or df.empty:
        return False
    last_ts = df["timestamp"].iloc[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)
    fresh = check_data_freshness(last_ts, max_age_minutes=30)
    price_cols = [
        c
        for c in df.columns
        if "price" in c.lower() or c.lower() in {"close", "open", "high", "low"}
    ]
    return fresh and not df[price_cols].isnull().any().any()


__all__ = [
    "check_data_freshness",
    "get_stale_symbols",
    "validate_trading_data",
    "emergency_data_check",
]
