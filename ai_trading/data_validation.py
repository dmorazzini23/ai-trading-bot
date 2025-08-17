"""Lightweight data validation helpers used by tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone

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
    symbol: str,
    *,
    max_staleness_minutes: int = 15,
) -> dict[str, float | str | bool]:
    """Return freshness info for ``symbol``."""  # AI-AGENT-REF
    try:
        last_ts = df.index[-1]
        if not isinstance(last_ts, datetime):
            raise TypeError
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_ts.astimezone(timezone.utc)
        minutes = age.total_seconds() / 60.0
        return {
            "symbol": symbol,
            "is_fresh": minutes <= max_staleness_minutes,
            "minutes_stale": minutes,
        }
    except Exception:
        return {"symbol": symbol, "is_fresh": False, "minutes_stale": float("inf")}


def get_stale_symbols(
    data_map: Mapping[str, Mapping[str, object]],
    *,
    max_staleness_minutes: int = 15,
) -> list[str]:
    out: list[str] = []
    for sym, info in (data_map or {}).items():
        fresh = False
        if isinstance(info, Mapping):
            fresh = bool(info.get("trading_ready", info.get("is_fresh")))
        else:
            fresh = check_data_freshness(info, sym, max_staleness_minutes=max_staleness_minutes)[
                "is_fresh"
            ]
        if not fresh:
            out.append(sym)
    return out


def validate_trading_data(
    data_map: Mapping[str, pd.DataFrame],
    *,
    max_staleness_minutes: int = 15,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for sym, df in (data_map or {}).items():
        info = check_data_freshness(df, sym, max_staleness_minutes=max_staleness_minutes)
        info["trading_ready"] = bool(info.get("is_fresh"))
        results[sym] = info
    return results


def emergency_data_check(
    symbols: Sequence[str],
    *,
    fetcher: Callable[..., pd.DataFrame] | None = None,
) -> bool:
    fetch = fetcher or get_bars
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=1)
    for sym in symbols:
        try:
            df = fetch(sym, start, end)
            if df is None or df.empty:
                return False
        except Exception:
            return False
    return True
