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
    "is_valid_ohlcv",
]


REQUIRED_PRICE_COLS = ("open", "high", "low", "close", "volume")


def is_valid_ohlcv(df: pd.DataFrame, min_rows: int = 50) -> bool:
    """Return True if ``df`` has required OHLCV columns and rows."""
    if df is None or df.empty:
        return False
    if not set(REQUIRED_PRICE_COLS).issubset(df.columns):
        return False
    return len(df) >= min_rows


def check_data_freshness(
    df: pd.DataFrame | None, symbol: str, *, max_staleness_minutes: int = 15
) -> dict[str, float | str | bool]:
    """Return freshness info for ``symbol`` handling empty/naive data."""
    if df is None or getattr(df, "empty", True):
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
    data_map: Mapping[str, object] | None, *, max_staleness_minutes: int = 15
) -> list[str]:
    """Return symbols whose data is stale.

    Accepts either ``{sym: DataFrame}`` or ``{sym: info mapping}``.
    """  # AI-AGENT-REF: accepts legacy shapes
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
    """Return mapping of freshness and trading readiness."""
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
    symbols_or_df: Sequence[str] | str | pd.DataFrame,
    symbol: str | None = None,
    *,
    fetcher: Callable[[str, str, datetime, datetime], pd.DataFrame] | None = None,
) -> bool:
    """Return True if any symbol yields non-empty recent bars.

    Back-compat: ``emergency_data_check(df, "AAPL")`` returns ``not df.empty``.
    """  # AI-AGENT-REF: legacy support
    if isinstance(symbols_or_df, pd.DataFrame) and isinstance(symbol, str):
        return not symbols_or_df.empty

    if isinstance(symbols_or_df, (str, bytes)):
        to_check = [symbols_or_df]
    elif isinstance(symbols_or_df, Sequence):  # type: ignore[redundant-expr]
        to_check = list(symbols_or_df)
    else:
        to_check = [str(symbols_or_df)]

    fetch = fetcher or get_bars
    end = datetime.now(UTC)
    start = end - timedelta(minutes=1)
    for sym in to_check:
        try:
            df = fetch(sym, "1Min", start, end)
            if df is not None and not df.empty:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False
