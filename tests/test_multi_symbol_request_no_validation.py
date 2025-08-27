from __future__ import annotations

from datetime import UTC, datetime
import types

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data.bars import StockBarsRequest, safe_get_stock_bars
from ai_trading.core import bot_engine as be


def test_multi_symbol_stockbarsrequest_no_validation_error():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    req = StockBarsRequest(
        symbol_or_symbols=["SPY", "AAPL"],
        timeframe=be._parse_timeframe("1Day"),
        start=start,
        end=end,
        feed="sip",
    )

    class DummyClient:
        def get_stock_bars(self, request):  # pragma: no cover - simple stub
            idx = pd.date_range(start, periods=1, tz="UTC")
            data = {
                ("SPY", "open"): [1.0],
                ("SPY", "high"): [1.0],
                ("SPY", "low"): [1.0],
                ("SPY", "close"): [1.0],
                ("SPY", "volume"): [1],
                ("AAPL", "open"): [1.0],
                ("AAPL", "high"): [1.0],
                ("AAPL", "low"): [1.0],
                ("AAPL", "close"): [1.0],
                ("AAPL", "volume"): [1],
            }
            df = pd.DataFrame(data, index=idx)
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            return types.SimpleNamespace(df=df)

    df = safe_get_stock_bars(DummyClient(), req, symbol="SPY", context="TEST")
    assert isinstance(df, pd.DataFrame)
