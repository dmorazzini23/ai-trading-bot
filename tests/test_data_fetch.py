from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from ai_trading.alpaca_api import get_bars_df  # AI-AGENT-REF


def test_get_bars_df_spy_day(monkeypatch):
    """get_bars_df should normalize SDK response data into a DataFrame."""
    pd = pytest.importorskip("pandas")

    class DummyStockBarsRequest:
        def __init__(
            self,
            *,
            symbol_or_symbols,
            timeframe,
            start,
            end,
            adjustment,
            feed,
        ):
            self.symbol_or_symbols = symbol_or_symbols
            self.timeframe = timeframe
            self.start = start
            self.end = end
            self.adjustment = adjustment
            self.feed = feed

    class DummyRest:
        def get_stock_bars(self, req):
            frame = pd.DataFrame(
                {
                    "open": [1.0],
                    "high": [1.1],
                    "low": [0.9],
                    "close": [1.05],
                    "volume": [100],
                },
                index=pd.DatetimeIndex(
                    [datetime(2024, 1, 2, tzinfo=UTC)],
                    name="timestamp",
                ),
            )
            return SimpleNamespace(df=frame)

    monkeypatch.setattr(
        "ai_trading.alpaca_api.get_stock_bars_request_cls",
        lambda: DummyStockBarsRequest,
    )
    monkeypatch.setattr(
        "ai_trading.alpaca_api._get_rest",
        lambda bars=True: DummyRest(),
    )

    df = get_bars_df("SPY", "1Day")
    assert len(df) > 0
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns


