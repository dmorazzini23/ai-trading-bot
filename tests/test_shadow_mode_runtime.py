import importlib
import sys
import types
from typing import Any, cast


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
