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


def _ts_utc_now() -> datetime:
    return datetime.now(UTC)


def check_data_freshness(
    df: pd.DataFrame,
    symbol: str,
    max_staleness_minutes: int = 15,
) -> bool:
    """Return True if ``df`` has data within ``max_staleness_minutes``."""
    if df is None or df.empty:
        return False
    ts = df.index.max()
    try:
        ts = pd.to_datetime(ts, utc=True).to_pydatetime()
    except Exception:  # noqa: BLE001
        return False
    return (_ts_utc_now() - ts) <= timedelta(minutes=int(max_staleness_minutes))


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
    max_staleness_minutes: int = 5,
) -> list[str]:
    """Return symbols whose data is older than ``max_staleness_minutes``."""
    out: list[str] = []
    for sym, df in frames.items():
        if not check_data_freshness(
            df, sym, max_staleness_minutes=max_staleness_minutes
        ):
            out.append(sym)
    return out


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
        except Exception:  # noqa: BLE001
            return False
    return True
