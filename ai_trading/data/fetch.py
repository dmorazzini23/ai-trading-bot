from __future__ import annotations

"""Compatibility wrappers for legacy data fetcher APIs.

This module exposes a minimal subset of the old ``ai_trading.data_fetcher``
interfaces so callers can transition to the ``ai_trading.data`` package without
importing the deprecated module directly. Each function delegates to
``ai_trading.data_fetcher`` at call time so test monkeypatches remain effective.
"""

from datetime import datetime
from typing import Any, Dict, List

from ai_trading import data_fetcher as _df

DataFetchError = _df.DataFetchError
DataFetchException = _df.DataFetchException
FinnhubAPIException = getattr(_df, "FinnhubAPIException", Exception)


def get_bars(
    symbol: str,
    timeframe: str,
    start: Any,
    end: Any,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
):
    return _df.get_bars(symbol, timeframe, start, end, feed=feed, adjustment=adjustment)


def get_bars_batch(
    symbols: List[str],
    timeframe: str,
    start: Any,
    end: Any,
    *,
    feed: str | None = None,
    adjustment: str | None = None,
) -> Dict[str, Any]:
    return _df.get_bars_batch(
        symbols, timeframe, start, end, feed=feed, adjustment=adjustment
    )


def get_minute_df(symbol: str, start: Any, end: Any, feed: str | None = None):
    return _df.get_minute_df(symbol, start, end, feed=feed)


def get_daily_df(symbol: str, start: Any, end: Any, feed: str | None = None):
    return get_bars(symbol, "1Day", start, end, feed=feed)


def build_fetcher(config: Any):
    return _df.build_fetcher(config)


def get_cached_minute_timestamp(symbol: str) -> int | None:
    return _df.get_cached_minute_timestamp(symbol)


def set_cached_minute_timestamp(symbol: str, ts_epoch_s: int) -> None:
    _df.set_cached_minute_timestamp(symbol, ts_epoch_s)


def clear_cached_minute_timestamp(symbol: str) -> None:
    _df.clear_cached_minute_timestamp(symbol)


def age_cached_minute_timestamps(max_age_seconds: int) -> int:
    return _df.age_cached_minute_timestamps(max_age_seconds)


def last_minute_bar_age_seconds(symbol: str) -> int | None:
    return _df.last_minute_bar_age_seconds(symbol)


def _build_daily_url(symbol: str, start: datetime, end: datetime) -> str:
    start_s = int(start.timestamp())
    end_s = int(end.timestamp())
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?period1={start_s}&period2={end_s}&interval=1d"
    )


__all__ = [
    "DataFetchError",
    "DataFetchException",
    "FinnhubAPIException",
    "get_bars",
    "get_bars_batch",
    "get_minute_df",
    "get_daily_df",
    "build_fetcher",
    "get_cached_minute_timestamp",
    "set_cached_minute_timestamp",
    "clear_cached_minute_timestamp",
    "age_cached_minute_timestamps",
    "last_minute_bar_age_seconds",
    "_build_daily_url",
]
