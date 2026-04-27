from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

"""Cache wrapper for daily bar fetches with network fallback.

This module wraps :func:`ai_trading.data.fetch.get_daily_df` with an in-memory
cache.  Each call first tries to fetch fresh data from the network.  If that
fails and a cached result exists, the cached data is returned and a warning is
emitted instead of raising an exception.  Successful fetches update the cache,
which is keyed by the function arguments.
"""

from datetime import UTC, datetime, timedelta
from typing import Any, Hashable, TYPE_CHECKING

import warnings
if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

from ai_trading.data.fetch import (
    DataFetchError,
    get_daily_df as _fetch_daily_df,
)

DAILY_CACHE_FALLBACK_EXCEPTIONS: tuple[type[Exception], ...] = (
    DataFetchError,
    TimeoutError,
    ConnectionError,
    OSError,
)
_CACHE_MAX_AGE = timedelta(days=1)

# Global in-memory cache mapping parameter tuples to DataFrames
_CACHE: dict[tuple[Hashable, ...], "pd.DataFrame | None"] = {}
_CACHE_TS: dict[tuple[Hashable, ...], datetime] = {}


def _timestamp_series(df: "pd.DataFrame") -> Any:
    import pandas as pd  # type: ignore

    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return pd.to_datetime(df.index, utc=True, errors="coerce")


def _cached_frame_usable(df: "pd.DataFrame | None", *, start: Any | None, fetched_at: datetime | None) -> bool:
    if df is None or getattr(df, "empty", True):
        return False
    if fetched_at is None or datetime.now(UTC) - fetched_at > _CACHE_MAX_AGE:
        return False
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return False
    try:
        close = df["close"]
        if close.isna().all():
            return False
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    try:
        ts = _timestamp_series(df).dropna()
    except (AttributeError, TypeError, ValueError):
        return False
    if ts.empty:
        return False
    if start is not None:
        import pandas as pd  # type: ignore

        start_ts = pd.to_datetime(start, utc=True, errors="coerce")
        if pd.notna(start_ts) and ts.max() < start_ts:
            return False
    return True


def get_daily_df(
    symbol: str,
    start: Any | None = None,
    end: Any | None = None,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
) -> "pd.DataFrame | None":
    """Fetch daily bars with cache fallback.

    Parameters mirror :func:`ai_trading.data.fetch.get_daily_df`.  Fresh data is
    requested from the network each call.  If the fetch fails and the result was
    previously cached, the cached DataFrame is returned with a warning instead
    of raising an error.
    """

    key: tuple[Hashable, ...] = (symbol, start, end, feed, adjustment)
    cached = _CACHE.get(key)
    cached_at = _CACHE_TS.get(key)
    try:
        df = _fetch_daily_df(symbol, start, end, feed=feed, adjustment=adjustment)
    except DAILY_CACHE_FALLBACK_EXCEPTIONS as exc:  # noqa: BLE001 - broad fallback for network issues
        if _cached_frame_usable(cached, start=start, fetched_at=cached_at):
            warnings.warn(
                f"Using cached daily data for {symbol} after fetch failure: {exc}",
                stacklevel=2,
            )
            return cached
        raise DataFetchError(str(exc)) from exc
    if _cached_frame_usable(df, start=start, fetched_at=datetime.now(UTC)):
        _CACHE[key] = df
        _CACHE_TS[key] = datetime.now(UTC)
    return df


__all__ = ["get_daily_df", "_CACHE"]
