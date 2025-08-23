import pytest

pytest.importorskip("alpaca_trade_api", reason="vendor SDK not installed")


def test_clients_built_once(monkeypatch):
    import ai_trading.core.bot_engine as be

    calls = {"trade": 0, "data": 0}

    class _T:
        pass

    class _D:
        pass

    def fake_trading_client(*a, **k):
        calls["trade"] += 1
        return _T()

    def fake_rest(*a, **k):
        calls["data"] += 1
        return _D()

    monkeypatch.setenv("ALPACA_API_KEY", "x")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "y")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    monkeypatch.setattr(
        be,
        "TradingClient",
        fake_trading_client,
        raising=False,
    )
    import alpaca_trade_api
    monkeypatch.setattr(alpaca_trade_api, "REST", fake_rest, raising=True)

    engine = be.BotEngine()
    _ = engine.trading_client
    _ = engine.trading_client
    _ = engine.data_client
    _ = engine.data_client

    assert calls["trade"] == 1
    assert calls["data"] == 1
