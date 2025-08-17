"""Lightweight data fetching helpers with patchable client."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

try:  # pragma: no cover - pandas optional in some tests
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

BAR_TIMEFRAME_DEFAULT = "1Min"
MAX_RETRIES_DEFAULT = 3
RETRYABLE_MESSAGES = ("Invalid format for parameter start",)

__all__ = [
    "ensure_datetime",
    "get_minute_df",
    "get_bars",
    "get_historical_data",
    "BAR_TIMEFRAME_DEFAULT",
    "MAX_RETRIES_DEFAULT",
    "RETRYABLE_MESSAGES",
    "_DATA_CLIENT",
    "_MINUTE_CACHE",
    "get_cached_minute_timestamp",
    "DataFetchError",
    "DataFetchException",
]

_DATA_CLIENT: Any | None = None
_MINUTE_CACHE: dict[str, tuple[datetime, pd.DataFrame]] = {}


class DataFetchError(Exception):
    """Raised when data fetching ultimately fails."""


DataFetchException = DataFetchError


def ensure_datetime(dt: datetime | str | pd.Timestamp) -> datetime:
    """Return a timezone-aware UTC ``datetime`` for ``dt``."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    elif "Timestamp" in type(dt).__name__:
        dt = dt.to_pydatetime()
    if not isinstance(dt, datetime):  # pragma: no cover - defensive
        raise TypeError(f"unsupported datetime value: {dt!r}")
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def get_minute_df(
    symbol: str,
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    *,
    timeframe: str = BAR_TIMEFRAME_DEFAULT,
    max_retries: int = MAX_RETRIES_DEFAULT,
    sleep_s: float = 0.1,
) -> pd.DataFrame:
    """Fetch minute bars for ``symbol`` with simple retry logic."""
    if _DATA_CLIENT is None:  # pragma: no cover - misconfigured in tests
        raise RuntimeError("_DATA_CLIENT not configured")
    start_dt = ensure_datetime(start)
    end_dt = ensure_datetime(end)
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return _DATA_CLIENT.get_stock_bars(
                symbol,
                start=start_dt,
                end=end_dt,
                timeframe=timeframe,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            if not any(m in msg for m in RETRYABLE_MESSAGES) or attempt >= max_retries - 1:
                raise
            time.sleep(sleep_s)
    raise last_exc or RuntimeError("get_minute_df failed")


def get_bars(
    symbol: str,
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    *,
    timeframe: str = BAR_TIMEFRAME_DEFAULT,
    max_retries: int = MAX_RETRIES_DEFAULT,
    sleep_s: float = 0.1,
) -> pd.DataFrame:
    """Alias for :func:`get_minute_df` for backward compatibility."""
    return get_minute_df(
        symbol,
        start,
        end,
        timeframe=timeframe,
        max_retries=max_retries,
        sleep_s=sleep_s,
    )


def get_historical_data(
    symbols: list[str] | tuple[str, ...],
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    *,
    timeframe: str = BAR_TIMEFRAME_DEFAULT,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Fetch bars for each symbol and return mapping."""
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = get_minute_df(sym, start, end, timeframe=timeframe, **kwargs)
    return out


def get_cached_minute_timestamp(symbol: str) -> datetime | None:
    entry = _MINUTE_CACHE.get(symbol)
    return entry[0] if entry else None
