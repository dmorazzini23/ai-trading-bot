"""Stub request classes for tests.

These classes mimic a tiny subset of the real ``alpaca-py`` request
models.  Only the attributes and behaviour required by the tests are
implemented here.  The goal is to provide a light‑weight stand‑in that
resembles the public API of the SDK so higher level code can be exercised
without importing the real package.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Iterable, Optional, Union


class BaseTimeseriesDataRequest:
    """Base request capturing common time series parameters."""

    def __init__(
        self,
        symbol_or_symbols: Union[str, Iterable[str]],
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        currency: Optional[str] = None,
        sort: Optional[str] = None,
        **extra: Any,
    ) -> None:
        if start and start.tzinfo is not None:
            start = start.astimezone(UTC).replace(tzinfo=None)
        if end and end.tzinfo is not None:
            end = end.astimezone(UTC).replace(tzinfo=None)

        self.symbol_or_symbols = symbol_or_symbols
        self.start = start
        self.end = end
        self.limit = limit
        self.currency = currency
        self.sort = sort
        for k, v in extra.items():
            setattr(self, k, v)


class BaseBarsRequest(BaseTimeseriesDataRequest):
    """Base request for bar data types."""

    def __init__(self, symbol_or_symbols, timeframe, **kwargs):
        super().__init__(symbol_or_symbols, **kwargs)
        self.timeframe = timeframe


class StockBarsRequest(BaseBarsRequest):
    """Request model for retrieving equity bar data."""

    def __init__(
        self,
        symbol_or_symbols,
        timeframe,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        adjustment: Optional[str] = None,
        feed: Optional[str] = None,
        sort: Optional[str] = None,
        asof: Optional[str] = None,
        currency: Optional[str] = None,
        **extra: Any,
    ) -> None:
        super().__init__(
            symbol_or_symbols,
            timeframe,
            start=start,
            end=end,
            limit=limit,
            currency=currency,
            sort=sort,
            **extra,
        )
        self.adjustment = adjustment
        self.feed = feed
        self.asof = asof


class StockLatestQuoteRequest:
    def __init__(self, *a, **k):  # pragma: no cover - trivial stub
        pass


__all__ = [
    "BaseTimeseriesDataRequest",
    "BaseBarsRequest",
    "StockBarsRequest",
    "StockLatestQuoteRequest",
]
