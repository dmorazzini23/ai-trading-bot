from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ai_trading.data.fetch.normalize import normalize_ohlcv_df


@dataclass(slots=True)
class _MemoEntry:
    ts: float
    key: Tuple[str, str, str, str]
    df: Any


_MEMO: Dict[Tuple[str, str, str, str], _MemoEntry] = {}
_TTL_S = 60.0


def get_daily_df_memoized(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    df_or_factory: Any,
) -> Any:
    """Return cached daily DataFrame within TTL, applying normalization once."""

    key = (symbol, timeframe, start, end)
    now = time.time()
    entry = _MEMO.get(key)
    if entry and (now - entry.ts) < _TTL_S:
        return entry.df

    if callable(df_or_factory):
        try:
            df = df_or_factory(symbol, timeframe, start, end)
        except TypeError:
            df = df_or_factory()
    else:
        df = df_or_factory

    try:
        normalized = normalize_ohlcv_df(df)
    except Exception:
        normalized = df

    _MEMO[key] = _MemoEntry(now, key, normalized)
    return normalized


def clear_memo() -> None:
    """Reset memoized values; primarily for tests."""

    _MEMO.clear()


__all__ = ["get_daily_df_memoized", "clear_memo"]
