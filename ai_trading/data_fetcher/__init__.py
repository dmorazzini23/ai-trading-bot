from __future__ import annotations

from datetime import UTC, datetime, timezone
from datetime import datetime, timezone
from time import sleep
from typing import Any, Dict, List

__all__ = [
    "ensure_datetime",
    "get_minute_df",
    "get_historical_data",
    "_DATA_CLIENT",
    "MAX_RETRIES",
]

# Tests patch this in place; keep the exact name.
_DATA_CLIENT = None
MAX_RETRIES = 3


def ensure_datetime(dt: Any) -> datetime:
    """Always return a timezone-aware UTC datetime (RFC3339-friendly)."""
    if isinstance(dt, datetime):
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    raise TypeError("ensure_datetime expects a datetime")


def get_minute_df(symbol: str, start: datetime, end: datetime):
    """Bounded retry specifically for datetime format errors."""
    assert _DATA_CLIENT is not None, "_DATA_CLIENT not configured"
    for i in range(MAX_RETRIES):
        try:
            return _DATA_CLIENT.get_stock_bars(
                symbol,
                start=ensure_datetime(start),
                end=ensure_datetime(end),
                timeframe="1Min",
            )
        except Exception as e:
            msg = str(e)
            if "Invalid format for parameter start" in msg:
                if i == MAX_RETRIES - 1:
                    raise
                sleep(0.05 * (2**i))
            else:
                raise


def get_historical_data(
    symbols: list[str], start: datetime, end: datetime
) -> dict[str, Any]:
    """Simple multi-symbol daily fetch used by tests."""
    assert _DATA_CLIENT is not None, "_DATA_CLIENT not configured"
    out: dict[str, Any] = {}
    for s in symbols:
        out[s] = _DATA_CLIENT.get_stock_bars(
            s,
            start=ensure_datetime(start),
            end=ensure_datetime(end),
            timeframe="1Day",
        )
    return out


# Compatibility helpers for modules expecting the full fetcher


def get_bars(*args, **kwargs):  # pragma: no cover - passthrough stub
    assert _DATA_CLIENT is not None, "_DATA_CLIENT not configured"
    return _DATA_CLIENT.get_stock_bars(*args, **kwargs)


def get_last_available_bar(symbol: str):  # pragma: no cover - passthrough stub
    assert _DATA_CLIENT is not None, "_DATA_CLIENT not configured"
    return _DATA_CLIENT.get_stock_bars(symbol, limit=1)


try:  # expose full data_fetcher when available
    from .full import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass
