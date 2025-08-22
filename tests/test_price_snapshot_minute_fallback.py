from types import SimpleNamespace

import pandas as pd

from ai_trading.portfolio import core as portfolio_core


def test_price_snapshot_minute_fallback(monkeypatch):
    ctx = SimpleNamespace(
        data_fetcher=SimpleNamespace(get_daily_df=lambda ctx, s: pd.DataFrame()),
        data_client=object(),
    )

    def fake_safe_get_stock_bars(client, request, symbol, context=""):
        return pd.DataFrame({"timestamp": [pd.Timestamp.utcnow()], "close": [123.0]})

    class DummyRequest:
        def __init__(self, *args, **kwargs):
            self.feed = kwargs.get("feed")
            self.timeframe = kwargs.get("timeframe")

    monkeypatch.setattr(portfolio_core, "safe_get_stock_bars", fake_safe_get_stock_bars)
    monkeypatch.setattr(portfolio_core, "StockBarsRequest", DummyRequest)
    monkeypatch.setattr(portfolio_core, "TimeFrame", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_core, "TimeFrameUnit", SimpleNamespace(Minute=None))

    price = portfolio_core.get_latest_price(ctx, "SPY")
    assert price == 123.0

