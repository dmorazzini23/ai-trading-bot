from __future__ import annotations

"""Cache wrapper for daily bar fetches with network fallback.

This module wraps :func:`ai_trading.data.fetch.get_daily_df` with an in-memory
cache.  Each call first tries to fetch fresh data from the network.  If that
fails and a cached result exists, the cached data is returned and a warning is
emitted instead of raising an exception.  Successful fetches update the cache,
which is keyed by the function arguments.
"""

from typing import Any, Hashable

import warnings
import pandas as pd

from ai_trading.data.fetch import (
    DataFetchError,
    get_daily_df as _fetch_daily_df,
)

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
    """Fetch daily bars with cache fallback.

    Parameters mirror :func:`ai_trading.data.fetch.get_daily_df`.  Fresh data is
    requested from the network each call.  If the fetch fails and the result was
    previously cached, the cached DataFrame is returned with a warning instead
    of raising an error.
    """

    key: tuple[Hashable, ...] = (symbol, start, end, feed, adjustment)
    cached = _CACHE.get(key)
    try:
        df = _fetch_daily_df(symbol, start, end, feed=feed, adjustment=adjustment)
    except Exception as exc:  # noqa: BLE001 - broad fallback for network issues
        if cached is not None:
            warnings.warn(
                f"Using cached daily data for {symbol} after fetch failure: {exc}",
                stacklevel=2,
            )
            return cached
        raise DataFetchError(str(exc)) from exc
    _CACHE[key] = df
    return df


__all__ = ["get_daily_df", "_CACHE"]
