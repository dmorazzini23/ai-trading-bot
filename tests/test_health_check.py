import builtins
import sys
import types

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


def _install_alpaca_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    key: str | None = "key",
    secret: str | None = "secret",
    base_url: str = "https://paper-api.alpaca.markets",
    client: object | None = None,
) -> None:
    """Install lightweight Alpaca stubs to avoid heavyweight imports."""

    stub_alpaca = types.ModuleType("ai_trading.alpaca_api")
    stub_alpaca.ALPACA_AVAILABLE = True
    monkeypatch.setitem(sys.modules, "ai_trading.alpaca_api", stub_alpaca)

    stub_bot = types.ModuleType("ai_trading.core.bot_engine")

    def _resolver():
        return key, secret, base_url

    stub_bot._resolve_alpaca_env = _resolver  # type: ignore[attr-defined]
    stub_bot.trading_client = client if client is not None else object()
    monkeypatch.setitem(sys.modules, "ai_trading.core.bot_engine", stub_bot)

    try:
        import ai_trading.core as core_pkg
    except Exception:
        pass
    else:
        monkeypatch.setattr(core_pkg, "bot_engine", stub_bot, raising=False)


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


def test_shadow_mode_disabled_when_credentials_missing(monkeypatch):
    import ai_trading.config.management as config_mgmt

    _install_alpaca_stubs(monkeypatch, key=None, secret=None, base_url="", client=None)
    monkeypatch.setattr(config_mgmt, "validate_required_env", lambda: {})
    monkeypatch.setattr(config_mgmt, "is_shadow_mode", lambda: True)

    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    _assert_payload_structure(data)
    assert data["ok"] is True
    assert data["alpaca"]["shadow_mode"] is False
    assert data["alpaca"]["has_key"] is False
    assert data["alpaca"]["has_secret"] is False


def test_jsonify_failure_preserves_ok_when_healthy(monkeypatch):
    import ai_trading.config.management as config_mgmt

    def broken_jsonify(payload):
        raise RuntimeError("json busted")

    _install_alpaca_stubs(
        monkeypatch,
        key="key",
        secret="secret",
        base_url="https://paper-api.alpaca.markets",
        client=object(),
    )
    monkeypatch.setattr(config_mgmt, "validate_required_env", lambda: {})
    monkeypatch.setattr(config_mgmt, "is_shadow_mode", lambda: False)
    monkeypatch.setattr(app_module, "jsonify", broken_jsonify, raising=False)

    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    _assert_payload_structure(data)
    assert data["ok"] is True
    assert data["alpaca"]["has_key"] is True
    assert data["alpaca"]["has_secret"] is True
    assert data["alpaca"]["shadow_mode"] is False
    _assert_error_contains(data, "json busted")
