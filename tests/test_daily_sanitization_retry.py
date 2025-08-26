from __future__ import annotations


import ai_trading.core.bot_engine as be_mod
import pytest
pd = pytest.importorskip("pandas")
from ai_trading.core.bot_engine import DataFetcher


def test_daily_retry_handles_callable(monkeypatch):
    # AI-AGENT-REF: ensure callable retry sanitization
    calls = {"n": 0}
    class Dummy:
        alpaca_api_key = "k"
        alpaca_secret_key_plain = "s"

    monkeypatch.setattr(be_mod, "get_settings", lambda: Dummy())

    def fake_safe_get_stock_bars(client, request, symbol, context):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("datetime argument was callable")
        idx = pd.to_datetime(["2025-08-19T13:30:00Z"])
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [0],
                "symbol": [symbol],
            },
            index=idx,
        )
        return df

    monkeypatch.setattr(be_mod, "safe_get_stock_bars", fake_safe_get_stock_bars)

    df = DataFetcher().get_daily_df(None, "SPY")
    assert calls["n"] == 2
    assert df is not None
