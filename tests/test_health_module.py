import importlib
import sys
from types import SimpleNamespace


def test_default_config_and_ctx_guard(monkeypatch):
    importlib.reload(sys.modules.get("ai_trading.health", importlib.import_module("ai_trading.health")))
    hc_mod = sys.modules["ai_trading.health"]
    hc = hc_mod.HealthCheck()
    routes = getattr(hc.app, "_routes", {})
    assert ("/healthz", "GET") in routes
    assert ("/metrics", "GET") in routes
    hc.run()
    monkeypatch.delitem(sys.modules, "ai_trading.health", raising=False)


def test_custom_config_applied(monkeypatch):
    importlib.reload(sys.modules.get("ai_trading.health", importlib.import_module("ai_trading.health")))
    hc_mod = sys.modules["ai_trading.health"]
    hc = hc_mod.HealthCheck(config={"DEBUG": True})
    assert hc.app.config["DEBUG"] is True
    routes = getattr(hc.app, "_routes", {})
    assert ("/healthz", "GET") in routes
    assert ("/metrics", "GET") in routes
    monkeypatch.delitem(sys.modules, "ai_trading.health", raising=False)


def test_healthcheck_uses_canonical_health_only_app(monkeypatch):
    importlib.reload(sys.modules.get("ai_trading.health", importlib.import_module("ai_trading.health")))
    hc_mod = sys.modules["ai_trading.health"]
    import ai_trading.app as app_module

    hc = hc_mod.HealthCheck(ctx=SimpleNamespace(service="ai-trading"))
    canonical_app = app_module.build_standalone_healthcheck_app(
        fail_fast_env=False,
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
    )

    assert set(getattr(hc.app, "_routes", {})) == set(getattr(canonical_app, "_routes", {}))
    monkeypatch.delitem(sys.modules, "ai_trading.health", raising=False)


def test_healthcheck_run_disables_reloader_and_enables_threading(monkeypatch):
    importlib.reload(sys.modules.get("ai_trading.health", importlib.import_module("ai_trading.health")))
    hc_mod = sys.modules["ai_trading.health"]
    hc = hc_mod.HealthCheck(ctx=SimpleNamespace(host="127.0.0.1", port=8081))
    called: dict[str, object] = {}

    def _fake_run(*, host, port, threaded=False, use_reloader=True, **kwargs):
        called["host"] = host
        called["port"] = port
        called["threaded"] = threaded
        called["use_reloader"] = use_reloader
        called["kwargs"] = kwargs

    import ai_trading.app as app_module

    monkeypatch.setattr(app_module, "suppress_flask_startup_noise", lambda: None)
    monkeypatch.setattr(hc.app, "run", _fake_run, raising=False)

    hc.run()

    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8081
    assert called["threaded"] is True
    assert called["use_reloader"] is False
    assert called["kwargs"] == {}
