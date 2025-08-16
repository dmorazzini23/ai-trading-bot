from __future__ import annotations

from datetime import UTC, datetime
from time import sleep
from typing import Any

__all__ = [
    "ensure_datetime",
    "get_minute_df",
    "get_historical_data",
    "get_bars",
    "get_last_available_bar",
    "_DATA_CLIENT",
    "MAX_RETRIES",
]

# Tests patch this in place.
_DATA_CLIENT = None  # type: ignore[var-annotated]
MAX_RETRIES = 3


def ensure_datetime(dt: Any) -> datetime:
    """Return timezone-aware UTC datetimes (Alpaca RFC3339 friendly)."""
    # AI-AGENT-REF: lightweight helper
    if isinstance(dt, datetime):
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise TypeError("ensure_datetime expects a datetime")


def get_minute_df(symbol: str, start: datetime, end: datetime):
    """Bounded retry; raises after MAX_RETRIES on persistent format errors."""
    # AI-AGENT-REF: retry loop
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
            if "Invalid format for parameter start" in msg and i < MAX_RETRIES - 1:
                sleep(0.05 * (2**i))
                continue
            raise


def get_historical_data(
    symbols: list[str], start: datetime, end: datetime
) -> dict[str, Any]:
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


def get_bars(*args, **kwargs):
    """Lightweight passthrough used by imports that expect full module."""  # AI-AGENT-REF: stub for compatibility
    assert _DATA_CLIENT is not None, "_DATA_CLIENT not configured"
    return _DATA_CLIENT.get_stock_bars(*args, **kwargs)


def get_last_available_bar(symbol: str):
    """Return the latest bar for a symbol."""  # AI-AGENT-REF: minimal stub
    assert _DATA_CLIENT is not None, "_DATA_CLIENT not configured"
    return _DATA_CLIENT.get_stock_bars(symbol, limit=1)


try:  # AI-AGENT-REF: expose full data_fetcher when available
    from .full import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass
