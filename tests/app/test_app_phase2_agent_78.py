from __future__ import annotations

import types

import pytest

import ai_trading.app as app_module


def test_resolve_standalone_healthcheck_port_rejects_shared_port() -> None:
    settings = types.SimpleNamespace(healthcheck_port=9001, api_port=9001)

    with pytest.raises(SystemExit, match="HEALTHCHECK_PORT to differ"):
        app_module._resolve_standalone_healthcheck_port(settings)


def test_run_standalone_healthcheck_app_logs_oserror() -> None:
    events: list[tuple[str, dict[str, object] | None]] = []

    class _App:
        def run(self, **_kwargs) -> None:
            raise OSError("address in use")

    logger = types.SimpleNamespace(
        warning=lambda event, extra=None: events.append((event, extra))
    )

    app_module.run_standalone_healthcheck_app(_App(), host="127.0.0.1", port=8081, logger=logger)

    assert events == [
        (
            "HEALTHCHECK_PORT_CONFLICT",
            {"host": "127.0.0.1", "port": 8081, "error": "address in use"},
        )
    ]


def test_run_standalone_healthcheck_app_raises_for_standalone_bind_error() -> None:
    events: list[tuple[str, dict[str, object] | None]] = []

    class _App:
        def run(self, **_kwargs) -> None:
            raise OSError("address in use")

    logger = types.SimpleNamespace(
        critical=lambda event, extra=None: events.append((event, extra)),
        warning=lambda event, extra=None: events.append((event, extra)),
    )

    with pytest.raises(SystemExit) as excinfo:
        app_module.run_standalone_healthcheck_app(
            _App(),
            host="127.0.0.1",
            port=8081,
            logger=logger,
            raise_on_bind_error=True,
        )

    assert excinfo.value.code == 1
    assert events == [
        (
            "HEALTHCHECK_PORT_CONFLICT",
            {"host": "127.0.0.1", "port": 8081, "error": "address in use"},
        )
    ]


def test_module_entrypoint_loads_dotenv_before_standalone_settings(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        app_module,
        "_managed_env",
        lambda name, default=None: "1" if name == "RUN_HEALTHCHECK" else default,
    )

    class _Settings:
        api_port = 9001
        healthcheck_port = 8081

    def _ensure_dotenv_loaded() -> None:
        calls.append("dotenv")

    def _get_settings() -> _Settings:
        calls.append("settings")
        return _Settings()

    monkeypatch.setattr("ai_trading.env.ensure_dotenv_loaded", _ensure_dotenv_loaded)
    monkeypatch.setattr("ai_trading.config.settings.get_settings", _get_settings)
    monkeypatch.setattr(
        app_module,
        "build_standalone_healthcheck_app",
        lambda **_kwargs: types.SimpleNamespace(logger=types.SimpleNamespace(info=lambda *_a, **_k: None)),
    )
    monkeypatch.setattr(
        app_module,
        "run_standalone_healthcheck_app",
        lambda *_args, **_kwargs: calls.append("run"),
    )

    app_module.run_module_entrypoint()

    assert calls == ["dotenv", "settings", "run"]


def test_parse_operator_token_map_accepts_json_and_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "_managed_env",
        lambda name, default=None: '{"Alice": "secret"}' if name == "AI_TRADING_OPERATOR_TOKEN_MAP" else default,
    )

    assert app_module._parse_operator_token_map() == {"alice": "secret"}

    monkeypatch.setattr(
        app_module,
        "_managed_env",
        lambda name, default=None: "Bob=token, Carol:other, broken" if name == "AI_TRADING_OPERATOR_TOKEN_MAP" else default,
    )

    assert app_module._parse_operator_token_map() == {"bob": "token", "carol": "other"}


def test_authenticate_operator_request_rejects_unknown_operator(monkeypatch) -> None:
    request = types.SimpleNamespace(
        headers={
            "Authorization": "Bearer secret",
            "X-AI-Trading-Operator-Id": "alice",
        }
    )
    monkeypatch.setattr(app_module, "request", request)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)
    monkeypatch.setattr(
        app_module,
        "_managed_env",
        lambda name, default=None: {"AI_TRADING_OPERATOR_TOKEN_MAP": '{"bob": "secret"}'}.get(
            name, default
        ),
    )

    operator_id, payload, status = app_module._authenticate_operator_request(scope="diag")

    assert operator_id is None
    assert status == 403
    assert payload == {"ok": False, "error": "operator is not authorized for this action"}


def test_create_health_only_app_wraps_non_mapping_payload(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_seed_pytest_env_defaults", lambda: None)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)
    monkeypatch.setattr(app_module, "_register_metrics_endpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        app_module,
        "build_canonical_healthz_payload",
        lambda **_kwargs: {"ok": False, "status": "degraded", "data": {"bad": object()}},
    )

    app = app_module.create_app(
        health_only=True,
        fail_fast_env=False,
        force_ok_for_pytest=False,
    )

    response = app.test_client().get("/healthz")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["service"] == "ai-trading"
    assert "alpaca" in payload
