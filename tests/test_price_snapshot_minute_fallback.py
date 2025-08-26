from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.portfolio import core as portfolio_core
from ai_trading import utils as _utils  # type: ignore


def test_price_snapshot_minute_fallback(monkeypatch):
    ctx = SimpleNamespace(
        data_fetcher=SimpleNamespace(get_daily_df=lambda ctx, s: pd.DataFrame()),
        data_client=object(),
    )

    def fake_safe_get_stock_bars(client, request, symbol, context=""):
        ts = _utils.time.utcnow()  # AI-AGENT-REF: avoid datetime.utcnow
        return pd.DataFrame({"timestamp": [pd.Timestamp(ts)], "close": [123.0]})

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
