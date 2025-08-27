from tests.optdeps import require
require("numpy")
require("pydantic")
import pytest
from ai_trading.risk.engine import RiskEngine
from ai_trading.settings import get_settings


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
    get_settings.cache_clear()

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
    get_settings.cache_clear()

    RiskEngine()

    assert calls.get("oauth_token") == "tok"
    assert "api_key" not in calls
    assert "secret_key" not in calls


def test_trading_client_conflicting_credentials(monkeypatch):
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_OAUTH", "tok")
    get_settings.cache_clear()

    with pytest.raises(RuntimeError):
        RiskEngine()


def _inject_dummy_trading(monkeypatch, cls):
    import types, sys

    mod_client = types.ModuleType("alpaca.trading.client")
    mod_client.TradingClient = cls
    mod_trading = types.ModuleType("alpaca.trading")
    mod_trading.client = mod_client
    monkeypatch.setitem(sys.modules, "alpaca", types.ModuleType("alpaca"))
    monkeypatch.setitem(sys.modules, "alpaca.trading", mod_trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", mod_client)


def test_get_rest_api_key_only(monkeypatch):
    calls: dict[str, str] = {}

    class Dummy:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    _inject_dummy_trading(monkeypatch, Dummy)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_OAUTH", raising=False)

    from ai_trading.alpaca_api import _get_rest

    _get_rest()
    assert calls.get("api_key") == "key"
    assert calls.get("secret_key") == "secret"
    assert "oauth_token" not in calls


def test_get_rest_oauth_only(monkeypatch):
    calls: dict[str, str] = {}

    class Dummy:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    _inject_dummy_trading(monkeypatch, Dummy)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("ALPACA_OAUTH", "tok")

    from ai_trading.alpaca_api import _get_rest

    _get_rest()
    assert calls.get("oauth_token") == "tok"
    assert "api_key" not in calls and "secret_key" not in calls


def test_get_rest_conflicting_credentials(monkeypatch):
    class Dummy:
        def __init__(self, **kwargs):
            pass

    _inject_dummy_trading(monkeypatch, Dummy)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_OAUTH", "tok")

    from ai_trading.alpaca_api import _get_rest

    with pytest.raises(RuntimeError):
        _get_rest()
