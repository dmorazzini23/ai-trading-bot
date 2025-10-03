import builtins
import json
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


def _assert_fallback_meta(payload: dict, *, used: bool, reasons: tuple[str, ...] = ()) -> None:
    meta = payload.get("meta", {})
    assert isinstance(meta, dict)
    fallback_meta = meta.get("fallback", {})
    assert isinstance(fallback_meta, dict)
    assert fallback_meta.get("used") is used
    meta_reasons = fallback_meta.get("reasons", [])
    assert isinstance(meta_reasons, list)
    if used or reasons:
        assert meta_reasons, "fallback metadata reasons should be present"
        for fragment in reasons:
            assert fragment in meta_reasons
    else:
        assert not meta_reasons


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
    _assert_fallback_meta(data, used=False)

    get_data = getattr(resp, "get_data", None)
    if callable(get_data):
        raw_body = get_data(as_text=True)
        if raw_body:
            raw_payload = json.loads(raw_body)
            _assert_payload_structure(raw_payload)
            _assert_fallback_meta(raw_payload, used=False)
            assert raw_payload == data
    assert data["ok"] is False
    assert data["alpaca"] == EXPECTED_ALPACA_MINIMAL
    _assert_error_contains(data, "boom")

    app.response_class = None
    handler = None
    view_functions = getattr(app, "view_functions", None)
    if isinstance(view_functions, dict):
        handler = view_functions.get("health")
    if handler is None:
        routes = getattr(app, "_routes", None)
        if isinstance(routes, dict):
            handler = routes.get("/health")
    assert handler is not None, "health handler should be registered"

    dict_payload = handler()
    assert isinstance(dict_payload, dict)
    _assert_payload_structure(dict_payload)
    _assert_fallback_meta(dict_payload, used=False)
    assert dict_payload["alpaca"] == EXPECTED_ALPACA_MINIMAL


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
    _assert_fallback_meta(data, used=True, reasons=("json busted", "RuntimeError"))
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
    _assert_fallback_meta(data, used=False)

    raw_body = resp.get_data(as_text=True)
    assert raw_body
    payload = json.loads(raw_body)
    _assert_payload_structure(payload)
    _assert_fallback_meta(payload, used=False)
    assert payload == data


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
    _assert_fallback_meta(data, used=True, reasons=("json busted", "RuntimeError"))
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
    _assert_fallback_meta(data, used=True, reasons=("jsonify unavailable",))
    assert data["ok"] is False
    _assert_error_contains(data, "jsonify unavailable")


def test_health_endpoint_missing_jsonify_dict_fallback_structure(monkeypatch):
    monkeypatch.delattr(app_module, "jsonify", raising=False)

    app = app_module.create_app()
    app.response_class = None

    handler = None
    view_functions = getattr(app, "view_functions", None)
    if isinstance(view_functions, dict):
        handler = view_functions.get("health")
    if handler is None:
        routes = getattr(app, "_routes", None)
        if isinstance(routes, dict):
            handler = routes.get("/health")
    assert handler is not None, "health handler should be registered"

    payload = handler()
    assert isinstance(payload, dict)
    _assert_payload_structure(payload)
    _assert_fallback_meta(payload, used=True, reasons=("jsonify unavailable",))
    assert payload["ok"] is False
    _assert_error_contains(payload, "jsonify unavailable")


def test_health_endpoint_handles_jsonify_import_error(monkeypatch):
    monkeypatch.setattr(app_module, "jsonify", None, raising=False)
    monkeypatch.setattr(app_module, "_jsonify_import_error", ImportError("flask missing"), raising=False)

    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    _assert_payload_structure(data)
    _assert_fallback_meta(
        data,
        used=True,
        reasons=("jsonify unavailable", "ImportError", "flask missing"),
    )
    assert data["ok"] is False
    _assert_error_contains(data, "jsonify unavailable", "ImportError", "flask missing")


def test_health_endpoint_dict_fallback_preserves_structure(monkeypatch):
    def broken_jsonify(payload):
        raise RuntimeError("json busted")

    monkeypatch.setattr(app_module, "jsonify", broken_jsonify, raising=False)

    app = app_module.create_app()
    app.response_class = None

    # When ``response_class`` is missing Flask cannot build a real ``Response``.
    # In these scenarios the view should expose the payload dictionary directly
    # so stub callers do not need to replicate Flask's response machinery.
    handler = None
    view_functions = getattr(app, "view_functions", None)
    if isinstance(view_functions, dict):
        handler = view_functions.get("health")
    if handler is None:
        routes = getattr(app, "_routes", None)
        if isinstance(routes, dict):
            handler = routes.get("/health")
    assert handler is not None, "health handler should be registered"

    payload = handler()
    assert isinstance(payload, dict)
    _assert_payload_structure(payload)
    _assert_fallback_meta(payload, used=True, reasons=("json busted", "RuntimeError"))
    assert payload["ok"] is False
    _assert_error_contains(payload, "json busted")


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
    _assert_fallback_meta(data, used=False)
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
    _assert_fallback_meta(data, used=True, reasons=("json busted", "RuntimeError"))
    assert data["ok"] is False
    assert data["alpaca"]["has_key"] is True
    assert data["alpaca"]["has_secret"] is True
    assert data["alpaca"]["shadow_mode"] is False
    _assert_error_contains(data, "json busted")


def test_json_dump_failure_rebuilds_payload(monkeypatch):
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

    original_dumps = app_module.json.dumps
    call_count = {"calls": 0}

    def flaky_dumps(payload, *args, **kwargs):
        call_count["calls"] += 1
        if call_count["calls"] == 1:
            raise TypeError("not serializable")
        return original_dumps(payload, *args, **kwargs)

    monkeypatch.setattr(app_module.json, "dumps", flaky_dumps)

    app = app_module.create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    _assert_payload_structure(data)
    _assert_fallback_meta(data, used=True, reasons=("not serializable",))
    assert data["ok"] is False
    assert data["alpaca"]["has_key"] is True
    assert data["alpaca"]["has_secret"] is True
    _assert_error_contains(data, "not serializable")
    assert call_count["calls"] >= 2
