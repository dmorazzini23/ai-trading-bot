import importlib
import sys
import types

import pytest

req_mod = types.ModuleType("requests")
req_mod.post = lambda *a, **k: None
req_mod.exceptions = types.SimpleNamespace(RequestException=Exception)
sys.modules.setdefault("requests", req_mod)

import ai_trading.shadow_mode.runtime  # registers lazy alpaca_api


def test_lazy_alpaca_api_behavior_switch(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)

    api = sys.modules["ai_trading.alpaca_api"]
    assert "submit_order" not in api.__dict__

    class Dummy:
        def submit_order(self, **kwargs):
            raise AssertionError("called real submit")

    monkeypatch.setenv("SHADOW_MODE", "1")
    res = api.submit_order("AAPL", "buy", qty=1, client=Dummy())
    assert res["id"].startswith("shadow-")

    monkeypatch.delenv("SHADOW_MODE", raising=False)
    importlib.reload(sys.modules["ai_trading.alpaca_api"])

    with pytest.raises(AssertionError):
        sys.modules["ai_trading.alpaca_api"].submit_order("AAPL", "buy", qty=1, client=Dummy())
