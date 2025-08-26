import importlib
import sys
import types

import pytest

req_mod = types.ModuleType("requests")
req_mod.post = lambda *a, **k: None
req_mod.exceptions = types.SimpleNamespace(RequestException=Exception)
sys.modules.setdefault("requests", req_mod)

import ai_trading.alpaca_api as api


def test_shadow_mode_env_change_affects_behavior(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE", raising=False)
    importlib.reload(api)

    class Dummy:
        def submit_order(self, **kwargs):
            raise AssertionError("called real submit")

    with pytest.raises(AssertionError):
        api.submit_order("AAPL", 1, "buy", client=Dummy())

    monkeypatch.setenv("SHADOW_MODE", "1")
    res = api.submit_order("AAPL", 1, "buy", client=Dummy())
    assert res["id"].startswith("shadow-")
