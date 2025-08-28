"""Unit tests for vendor Alpaca request stubs.

These tests ensure the lightweight stub classes mimic the real SDK's
behaviour when instantiated using positional or keyword arguments.
"""

from datetime import UTC, datetime

from tests.vendor_stubs.alpaca.data.requests import StockBarsRequest
from tests.vendor_stubs.alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def _tf_day():
    return TimeFrame(1, TimeFrameUnit.Day)


def test_stock_bars_request_positional_args():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    req = StockBarsRequest("SPY", _tf_day(), start=start, limit=10, feed="sip")

    assert req.symbol_or_symbols == "SPY"
    assert req.timeframe == _tf_day()
    assert req.start == start.astimezone(UTC).replace(tzinfo=None)
    assert req.limit == 10
    assert req.feed == "sip"


def test_stock_bars_request_keyword_args():
    start = datetime(2024, 2, 1, tzinfo=UTC)
    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=_tf_day(),
        start=start,
        adjustment="raw",
        asof="2024-02-01",
    )

    assert req.symbol_or_symbols == "SPY"
    assert req.timeframe == _tf_day()
    assert req.start == start.astimezone(UTC).replace(tzinfo=None)
    assert req.adjustment == "raw"
    assert req.asof == "2024-02-01"

