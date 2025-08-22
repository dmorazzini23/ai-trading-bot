from __future__ import annotations

import types

import pandas as pd
from ai_trading.core import bot_engine as be


def test_callable_triggers_single_debug(monkeypatch, caplog):
    caplog.set_level("DEBUG")

    class DummyReq:
        def __init__(self, **kwargs):
            self.start = lambda: kwargs.get("start")
            self.end = lambda: kwargs.get("end")
            self.symbol_or_symbols = kwargs.get("symbol_or_symbols")
            self.timeframe = kwargs.get("timeframe")
            self.feed = kwargs.get("feed")

    monkeypatch.setattr(be, "StockBarsRequest", DummyReq)
    monkeypatch.setattr(be, "is_market_open", lambda: True)

    class DummySettings:
        alpaca_api_key = "k"
        alpaca_secret_key_plain = "s"

    monkeypatch.setattr(be, "get_settings", lambda: DummySettings)
    monkeypatch.setattr(be, "StockHistoricalDataClient", lambda *a, **k: object())

    def fake_safe(client, req, symbol, ctx):
        return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2025-08-19", tz="UTC")])

    monkeypatch.setattr(be, "safe_get_stock_bars", fake_safe)

    fetcher = be.DataFetcher()
    ctx = types.SimpleNamespace()
    fetcher.get_daily_df(ctx, "SPY")

    records = [r for r in caplog.records if r.message == "DAILY_BARS_INPUT_SANITIZED"]
    assert len(records) == 1

