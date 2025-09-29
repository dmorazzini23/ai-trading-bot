import builtins

import pytest

import ai_trading.app as app_module


EXPECTED_ALPACA_MINIMAL = {
    "sdk_ok": False,
    "initialized": False,
    "client_attached": False,
    "has_key": False,
    "has_secret": False,
    "base_url": "",
    "paper": False,
    "shadow_mode": False,
}


def _assert_payload_structure(payload: dict) -> None:
    alpaca_keys = set(EXPECTED_ALPACA_MINIMAL)
    assert set(payload) >= {"ok", "alpaca"}
    assert isinstance(payload["ok"], bool)
    assert set(payload["alpaca"]) == alpaca_keys
    bool_keys = alpaca_keys - {"base_url"}
    assert all(isinstance(payload["alpaca"][key], bool) for key in bool_keys)
    assert isinstance(payload["alpaca"]["base_url"], str)


def _assert_error_contains(payload: dict, *expected: str) -> None:
    err = payload.get("error", "")
    assert isinstance(err, str) and err
    for fragment in expected:
        assert fragment in err


def test_health_endpoint_handles_import_error(monkeypatch):
    original_import = builtins.__import__

    def fail_alpaca(name, *args, **kwargs):
        if name == "ai_trading.alpaca_api":
            raise ImportError("boom")
        if name == "ai_trading.core.bot_engine":
            pytest.fail("_resolve_alpaca_env should not be touched when alpaca import fails")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_alpaca)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    _assert_payload_structure(data)
    assert data["ok"] is False
    assert data["alpaca"] == EXPECTED_ALPACA_MINIMAL
    _assert_error_contains(data, "boom")


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
    _assert_payload_structure(data)
    assert data["ok"] is False
    assert data["alpaca"] == EXPECTED_ALPACA_MINIMAL
    _assert_error_contains(data, "boom", "json busted")


def test_health_endpoint_structure_is_stable():
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    _assert_payload_structure(data)


def test_health_endpoint_jsonify_failure_uses_sanitized_payload(monkeypatch):
    def broken_jsonify(payload):
        raise RuntimeError("json busted")

    monkeypatch.setattr(app_module, "jsonify", broken_jsonify)

    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    _assert_payload_structure(data)
    assert data["ok"] is False
    _assert_error_contains(data, "json busted")


def test_health_endpoint_handles_missing_jsonify(monkeypatch):
    monkeypatch.delattr(app_module, "jsonify", raising=False)

    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    _assert_payload_structure(data)
    assert data["ok"] is False
    _assert_error_contains(data, "jsonify unavailable")
