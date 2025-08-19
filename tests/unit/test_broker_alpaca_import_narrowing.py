import importlib
import sys


def test_alpaca_import_without_requests(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    mod = importlib.reload(importlib.import_module("ai_trading.broker.alpaca"))
    assert hasattr(mod, "AlpacaBroker")
    assert "TradingClient" in mod.__all__
