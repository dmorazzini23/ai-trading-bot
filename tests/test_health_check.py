import builtins
import ai_trading.app as app_module


def test_health_endpoint_handles_import_error(monkeypatch):
    original_import = builtins.__import__

    def fail_alpaca(name, *args, **kwargs):
        if name == "ai_trading.alpaca_api":
            raise ImportError("boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_alpaca)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {
        "ok": False,
        "error": "boom",
        "alpaca": {
            "sdk_ok": False,
            "initialized": False,
            "client_attached": False,
            "has_key": False,
            "has_secret": False,
            "base_url": "",
            "paper": False,
            "shadow_mode": False,
        },
    }


def test_health_endpoint_returns_plain_dict_when_jsonify_fails(monkeypatch):
    original_import = builtins.__import__

    def fail_alpaca(name, *args, **kwargs):
        if name == "ai_trading.alpaca_api":
            raise ImportError("boom")
        return original_import(name, *args, **kwargs)

    def broken_jsonify(payload):
        raise RuntimeError("json busted")

    monkeypatch.setattr(builtins, "__import__", fail_alpaca)
    monkeypatch.setattr(app_module, "jsonify", broken_jsonify)

    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {
        "ok": False,
        "error": "boom",
        "alpaca": {
            "sdk_ok": False,
            "initialized": False,
            "client_attached": False,
            "has_key": False,
            "has_secret": False,
            "base_url": "",
            "paper": False,
            "shadow_mode": False,
        },
    }
