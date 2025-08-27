"""Backward compatibility rest stub mapping to alpaca-py data classes."""
from tests.vendor_stubs.alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from tests.vendor_stubs.alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
)

__all__ = [
    "TimeFrame",
    "TimeFrameUnit",
    "StockBarsRequest",
    "StockLatestQuoteRequest",
]
