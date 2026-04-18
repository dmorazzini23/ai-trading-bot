import importlib
import sys


def test_default_config_and_ctx_guard(monkeypatch):
    importlib.reload(sys.modules.get("ai_trading.health", importlib.import_module("ai_trading.health")))
    hc_mod = sys.modules["ai_trading.health"]
    hc = hc_mod.HealthCheck()
    assert hc.app.config == {}
    hc.run()
    monkeypatch.delitem(sys.modules, "ai_trading.health", raising=False)


def test_custom_config_applied(monkeypatch):
    importlib.reload(sys.modules.get("ai_trading.health", importlib.import_module("ai_trading.health")))
    hc_mod = sys.modules["ai_trading.health"]
    hc = hc_mod.HealthCheck(config={"DEBUG": True})
    assert hc.app.config["DEBUG"] is True
    monkeypatch.delitem(sys.modules, "ai_trading.health", raising=False)
