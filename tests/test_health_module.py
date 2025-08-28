import importlib
import sys
import types


def _install_flask_stub(monkeypatch):
    import sys
    stub = types.ModuleType("flask")

    class _Flask:
        def __init__(self, *a, **k):
            self.config = {}

        def route(self, *a, **k):
            def decorator(func):
                return func

            return decorator

        def run(self, *a, **k):
            pass

    stub.Flask = _Flask
    stub.jsonify = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "flask", stub)


def test_default_config_and_ctx_guard(monkeypatch):
    import importlib
    import sys
    _install_flask_stub(monkeypatch)
    importlib.reload(sys.modules.get("ai_trading.health", importlib.import_module("ai_trading.health")))
    hc_mod = sys.modules["ai_trading.health"]
    hc = hc_mod.HealthCheck()
    assert hc.app.config == {}
    hc.run()
    monkeypatch.delitem(sys.modules, "ai_trading.health", raising=False)


def test_custom_config_applied(monkeypatch):
    import importlib
    import sys
    _install_flask_stub(monkeypatch)
    importlib.reload(sys.modules.get("ai_trading.health", importlib.import_module("ai_trading.health")))
    hc_mod = sys.modules["ai_trading.health"]
    hc = hc_mod.HealthCheck(config={"DEBUG": True})
    assert hc.app.config["DEBUG"] is True
    monkeypatch.delitem(sys.modules, "ai_trading.health", raising=False)
