from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trading.core import bot_engine as be


def test_daily_request_sanitizes_inputs(monkeypatch):
    fetcher = be.DataFetcher()
    symbol = "SPY"

    # Skip market-open logic
    monkeypatch.setattr(be, "is_market_open", lambda: True)

    class DummySettings:
        alpaca_api_key = "k"
        alpaca_secret_key_plain = "s"

    monkeypatch.setattr(be, "get_settings", lambda: DummySettings)

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(be, "StockHistoricalDataClient", DummyClient)

    class DummyRequest:
        def __init__(self, symbol_or_symbols, timeframe, start, end, feed):
            self.symbol_or_symbols = symbol_or_symbols
            self.timeframe = timeframe
            # Store as mixed types to test sanitization
            self.start = start.strftime("%Y-%m-%d")  # str
            self.end = (end - timedelta(days=1)).date()  # date
            self.feed = feed

    monkeypatch.setattr(be, "StockBarsRequest", DummyRequest)

    calls = {"n": 0}

    def fake_safe_get_stock_bars(client, req, sym, ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("datetime argument was callable")
        assert isinstance(req.start, datetime) and req.start.tzinfo is UTC
        assert isinstance(req.end, datetime) and req.end.tzinfo is UTC
        return pd.DataFrame()  # empty => function returns None

    monkeypatch.setattr(be, "safe_get_stock_bars", fake_safe_get_stock_bars)

    result = fetcher.get_daily_df(object(), symbol)

    assert calls["n"] == 2
    assert result is None

