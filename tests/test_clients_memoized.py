import pytest

from tests.optdeps import require

require("requests")


def test_clients_built_once(monkeypatch):
    import ai_trading.core.bot_engine as be

    calls = {"trade": 0, "data": 0}

    class MockTradingClient:
        def __init__(self, *a, **k):
            calls["trade"] += 1

    class MockDataClient:
        def __init__(self, *a, **k):
            calls["data"] += 1

    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("WEBHOOK_SECRET", "s")
    monkeypatch.setenv("CAPITAL_CAP", "0.05")
    monkeypatch.setenv("DOLLAR_RISK_LIMIT", "0.05")

    engine = be.BotEngine(
        trading_client_cls=MockTradingClient, data_client_cls=MockDataClient
    )
    calls["trade"] = 0
    calls["data"] = 0
    tc1 = engine.trading_client
    tc2 = engine.trading_client
    dc1 = engine.data_client
    dc2 = engine.data_client

    assert tc1 is tc2
    assert dc1 is dc2
    assert calls["trade"] == 1
    assert calls["data"] == 1
