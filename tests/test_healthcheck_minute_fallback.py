import logging
from types import SimpleNamespace

import pandas as pd

from ai_trading.core import bot_engine


def test_healthcheck_minute_fallback(monkeypatch, caplog):
    ctx = SimpleNamespace()
    ctx.data_fetcher = SimpleNamespace(get_daily_df=lambda ctx, sym: pd.DataFrame())
    ctx.data_client = object()

    def fake_safe_get_stock_bars(client, request, symbol, context=""):
        rng = pd.date_range("2024-01-01", periods=390, freq="T", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": rng,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            }
        )

    class DummyRequest:
        def __init__(self, *args, **kwargs):
            self.feed = kwargs.get("feed")
            self.timeframe = kwargs.get("timeframe")

    monkeypatch.setattr(bot_engine, "safe_get_stock_bars", fake_safe_get_stock_bars)
    monkeypatch.setattr(bot_engine, "StockBarsRequest", DummyRequest)
    monkeypatch.setattr(bot_engine, "TimeFrame", lambda *a, **k: None)
    monkeypatch.setattr(bot_engine, "TimeFrameUnit", SimpleNamespace(Minute=None))
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)

    with caplog.at_level(logging.INFO):
        bot_engine.data_source_health_check(ctx, ["AAPL"])
    assert any("minute fallback ok" in r.message for r in caplog.records)
