from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    df: pd.DataFrame, symbol: str, *, max_staleness_minutes: int = 15
) -> dict[str, float | str | bool]:
    """Return freshness info for ``symbol`` (back-compat shape)."""
    if df is None or df.empty:
        return {"symbol": symbol, "is_fresh": False, "minutes_stale": float("inf")}
    try:
        last_ts = df.index[-1]
        if not isinstance(last_ts, datetime):
            last_ts = datetime.fromtimestamp(float(last_ts), tz=UTC)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)
        age = datetime.now(UTC) - last_ts.astimezone(UTC)
        minutes = age.total_seconds() / 60.0
        return {
            "symbol": symbol,
            "is_fresh": minutes <= max_staleness_minutes,
            "minutes_stale": minutes,
        }
    except Exception:  # noqa: BLE001
        return {"symbol": symbol, "is_fresh": False, "minutes_stale": float("inf")}


def get_stale_symbols(
    data_map: Mapping[str, object], *, max_staleness_minutes: int = 15
) -> list[str]:
    """Return symbols whose data is stale."""  # AI-AGENT-REF: accepts legacy shapes
    out: list[str] = []
    for sym, info in (data_map or {}).items():
        if isinstance(info, Mapping):
            fresh = bool(info.get("trading_ready", info.get("is_fresh")))
        else:
            fresh = bool(
                check_data_freshness(
                    info, sym, max_staleness_minutes=max_staleness_minutes
                )["is_fresh"]
            )
        if not fresh:
            out.append(sym)
    return out


def validate_trading_data(
    data_map: Mapping[str, pd.DataFrame] | None,
    *,
    max_staleness_minutes: int = 15,
) -> dict[str, dict[str, object]]:
    """Back-compat validation returning both is_fresh and trading_ready."""
    data_map = data_map or {}
    results: dict[str, dict[str, object]] = {}
    for sym, df in data_map.items():
        info = check_data_freshness(
            df, sym, max_staleness_minutes=max_staleness_minutes
        )
        info["trading_ready"] = bool(info.get("is_fresh"))
        results[sym] = info
    return results


def emergency_data_check(
    data_or_symbols: Sequence[str] | pd.DataFrame,
    symbol: str | None = None,
    fetcher: Callable[..., pd.DataFrame] | None = None,
) -> bool:
    """Flexible emergency data check for legacy call patterns."""
    if isinstance(data_or_symbols, pd.DataFrame):
        return data_or_symbols is not None and not data_or_symbols.empty
    symbols = list(data_or_symbols)
    fetch = fetcher or get_bars
    end = datetime.now(UTC)
    start = end - timedelta(minutes=1)
    for sym in symbols:
        try:
            # Try common call shapes without penalizing callers
            try:
                df = fetch(sym, "1Min", start, end)
            except TypeError:
                try:
                    df = fetch(sym, start, end)  # legacy shape
                except TypeError:
                    df = fetch(sym, start=start, end=end)
            if df is not None and not df.empty:
                return True
        except Exception:  # noqa: BLE001
            pass
    return False
