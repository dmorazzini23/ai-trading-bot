from tests.optdeps import require
require("numpy")
require("pydantic")
import pytest
from ai_trading.risk.engine import RiskEngine


def test_trading_client_api_key_only(monkeypatch):
    calls = {}

    class Dummy:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_OAUTH", raising=False)
    monkeypatch.setattr("ai_trading.risk.engine.TradingClient", Dummy)

    RiskEngine()

    assert calls.get("api_key") == "key"
    assert calls.get("secret_key") == "secret"
    assert "oauth_token" not in calls


def test_trading_client_oauth_only(monkeypatch):
    calls = {}

    class Dummy:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_OAUTH", "tok")
    monkeypatch.setattr("ai_trading.risk.engine.TradingClient", Dummy)

    RiskEngine()

    assert calls.get("oauth_token") == "tok"
    assert "api_key" not in calls
    assert "secret_key" not in calls


def test_trading_client_conflicting_credentials(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_OAUTH", "tok")

    with pytest.raises(RuntimeError):
        RiskEngine()
