import importlib
import sys
import types

import pytest

req_mod = types.ModuleType("requests")
req_mod.post = lambda *a, **k: None


class _RequestException(Exception):
    pass


req_mod.exceptions = types.SimpleNamespace(RequestException=_RequestException)
sys.modules.setdefault("requests", req_mod)

runtime_mod = importlib.import_module("ai_trading.shadow_mode.runtime")


def test_lazy_alpaca_api_placeholder_persists_and_respects_shadow(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)

    sys.modules.pop("ai_trading.alpaca_api", None)
    runtime = importlib.reload(runtime_mod)

    placeholder = runtime.ensure_alpaca_api()
    assert isinstance(placeholder, runtime._LazyModule)
    assert "submit_order" not in placeholder.__dict__

    # Load the real module to patch configuration behaviour while keeping the
    # placeholder registered.
    real_module = placeholder._load()
    assert sys.modules["ai_trading.alpaca_api"] is placeholder
    assert placeholder._real is real_module

    state: dict[str, bool] = {"shadow": True}

    def _from_env():
        return real_module._AlpacaConfig(
            "https://paper-api.alpaca.markets", None, None, state["shadow"]
        )

    monkeypatch.setattr(real_module._AlpacaConfig, "from_env", staticmethod(_from_env))

    class DummyClient:
        def __init__(self) -> None:
            self.calls = 0

        def submit_order(self, **kwargs):
            self.calls += 1
            raise AssertionError("called real submit")

    # In shadow mode the client must never be touched and the placeholder
    # remains active in ``sys.modules``.
    client_shadow = DummyClient()
    res = placeholder.submit_order("AAPL", "buy", qty=1, client=client_shadow)
    assert res["id"].startswith("shadow-")
    assert client_shadow.calls == 0
    assert sys.modules["ai_trading.alpaca_api"] is placeholder
    assert runtime.ensure_alpaca_api() is placeholder

    # Switching to non-shadow mode should call the real client while keeping the
    # lazy module registered for future imports.
    state["shadow"] = False
    client_real = DummyClient()
    with pytest.raises(AssertionError):
        placeholder.submit_order("AAPL", "buy", qty=1, client=client_real)
    assert client_real.calls == 1
    assert sys.modules["ai_trading.alpaca_api"] is placeholder
    assert runtime.ensure_alpaca_api() is placeholder


def test_lazy_alpaca_api_placeholder_reinstated_after_real_import(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)

    import importlib

    sys.modules.pop("ai_trading.alpaca_api", None)
    sys.modules.pop("ai_trading.shadow_mode.runtime", None)

    real_api = importlib.import_module("ai_trading.alpaca_api")
    assert sys.modules["ai_trading.alpaca_api"] is real_api

    # Importing the runtime should re-register the lazy placeholder even though the
    # real module is already loaded in ``sys.modules``.
    runtime = importlib.import_module("ai_trading.shadow_mode.runtime")

    placeholder = sys.modules["ai_trading.alpaca_api"]
    assert isinstance(placeholder, runtime._LazyModule)
    assert placeholder._real is real_api
