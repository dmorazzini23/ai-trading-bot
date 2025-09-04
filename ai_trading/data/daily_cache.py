from __future__ import annotations

"""Simple cache wrapper for daily bar fetches.

This module wraps :func:`ai_trading.data.fetch.get_daily_df` with an in-memory
cache.  Results from a fetch are stored in ``_CACHE`` keyed by the function
arguments so subsequent calls with the same parameters reuse the cached
DataFrame instead of hitting the data provider again.
"""

from typing import Any, Hashable

import pandas as pd

from ai_trading.data.fetch import get_daily_df as _fetch_daily_df

# Global in-memory cache mapping parameter tuples to DataFrames
_CACHE: dict[tuple[Hashable, ...], pd.DataFrame | None] = {}


def get_daily_df(
    symbol: str,
    start: Any | None = None,
    end: Any | None = None,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
) -> pd.DataFrame | None:
    """Fetch daily bars with a simple cache.

    Parameters mirror :func:`ai_trading.data.fetch.get_daily_df`.  The result is
    cached and returned directly on subsequent calls with the same arguments.
    """

    key: tuple[Hashable, ...] = (symbol, start, end, feed, adjustment)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    df = _fetch_daily_df(symbol, start, end, feed=feed, adjustment=adjustment)
    _CACHE[key] = df
    return df


__all__ = ["get_daily_df", "_CACHE"]
