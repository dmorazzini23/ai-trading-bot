import importlib
import sys
import types
from typing import Any, cast

import pytest

req_mod = types.ModuleType("requests")


class _RequestException(Exception):
    pass


cast(Any, req_mod).post = lambda *a, **k: None
cast(Any, req_mod).exceptions = types.SimpleNamespace(RequestException=_RequestException)
sys.modules.setdefault("requests", req_mod)


def test_runtime_import_does_not_register_lazy_alpaca_placeholder(monkeypatch):
    sys.modules.pop("ai_trading.alpaca_api", None)
    sys.modules.pop("ai_trading.shadow_mode.runtime", None)

    runtime = importlib.import_module("ai_trading.shadow_mode.runtime")

    assert "ai_trading.alpaca_api" not in sys.modules
    assert not hasattr(runtime, "_LazyModule")


def test_ensure_alpaca_api_returns_canonical_module(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    sys.modules.pop("ai_trading.alpaca_api", None)
    runtime = importlib.import_module("ai_trading.shadow_mode.runtime")

    alpaca_api = runtime.ensure_alpaca_api()

    assert alpaca_api is sys.modules["ai_trading.alpaca_api"]
    assert runtime.ensure_alpaca_api() is alpaca_api


def test_ensure_alpaca_api_preserves_shadow_submit_behavior(monkeypatch):
    sys.modules.pop("ai_trading.alpaca_api", None)
    runtime = importlib.import_module("ai_trading.shadow_mode.runtime")
    alpaca_api = runtime.ensure_alpaca_api()

    state: dict[str, bool] = {"shadow": True}

    def _from_env():
        return alpaca_api._AlpacaConfig(
            "https://paper-api.alpaca.markets", None, None, state["shadow"]
        )

    monkeypatch.setattr(alpaca_api._AlpacaConfig, "from_env", staticmethod(_from_env))

    class DummyClient:
        def __init__(self) -> None:
            self.calls = 0

        def submit_order(self, **kwargs):
            self.calls += 1
            raise AssertionError("called real submit")

    client_shadow = DummyClient()
    res = alpaca_api.submit_order("AAPL", "buy", qty=1, client=client_shadow)
    assert res["id"].startswith("shadow-")
    assert client_shadow.calls == 0

    state["shadow"] = False
    client_real = DummyClient()
    with pytest.raises(AssertionError):
        alpaca_api.submit_order("AAPL", "buy", qty=1, client=client_real)
    assert client_real.calls == 1
