from __future__ import annotations

from datetime import UTC, datetime

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data.bars import StockBarsRequest, TimeFrame, safe_get_stock_bars


def test_request_timestamps_sanitized_and_passed_to_get_bars():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    req = StockBarsRequest("SPY", TimeFrame.Day, start=start, end=end, feed="sip")

    captured: dict[str, str] = {}

    class DummyClient:
        def get_bars(self, symbol_or_symbols, timeframe, **params):
            captured.update({"start": params.get("start"), "end": params.get("end")})
            return pd.DataFrame()

    safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")

    expected_end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    assert req.start == start.isoformat()
    assert req.end == expected_end.isoformat()
    assert captured["start"] == start.isoformat()
    assert captured["end"] == expected_end.isoformat()


def test_request_timestamps_sanitized_for_get_stock_bars():
    start = datetime(2024, 1, 3, tzinfo=UTC)
    end = datetime(2024, 1, 4, tzinfo=UTC)
    req = StockBarsRequest("SPY", TimeFrame.Day, start=start, end=end, feed="sip")

    captured: dict[str, str] = {}

    class DummyClient:
        class Resp:
            df = pd.DataFrame()

        def get_stock_bars(self, request):  # pragma: no cover - simple stub
            captured["start"] = request.start
            captured["end"] = request.end
            return self.Resp()

    safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")

    expected_end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    assert captured["start"] == start.isoformat()
    assert captured["end"] == expected_end.isoformat()
