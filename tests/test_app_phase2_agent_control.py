from __future__ import annotations

import json
from types import SimpleNamespace

import ai_trading.app as app_mod
from ai_trading.app import create_app


def _headers(*, operator_id: str = "ops@example.com", token: str = "ops-token") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-AI-Trading-Operator-Id": operator_id,
    }


def test_operator_token_map_accepts_legacy_delimited_format(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv(
        "AI_TRADING_OPERATOR_TOKEN_MAP",
        "ops@example.com=ops-token, reader@example.com:reader-token, malformed",
    )

    app = create_app()
    client = app.test_client()

    response = client.get(
        "/operator/control-plane/services",
        headers=_headers(operator_id="reader@example.com", token="reader-token"),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["section"] == "services"


def test_operator_read_rejects_unknown_token_bound_operator(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv(
        "AI_TRADING_OPERATOR_TOKEN_MAP",
        json.dumps({"ops@example.com": "ops-token"}),
    )

    app = create_app()
    client = app.test_client()

    response = client.get(
        "/operator/control-plane",
        headers=_headers(operator_id="intruder@example.com", token="ops-token"),
    )

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "operator is not authorized for this action"


def test_control_plane_unknown_section_returns_404(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv(
        "AI_TRADING_OPERATOR_TOKEN_MAP",
        json.dumps({"ops@example.com": "ops-token"}),
    )

    class _FakeControlPlaneService:
        def __init__(self, *, service_name: str) -> None:
            assert service_name == "ai-trading"

        def section(self, section: str) -> dict[str, object]:
            assert section == "services"
            raise KeyError(section)

    monkeypatch.setattr(app_mod, "ControlPlaneService", _FakeControlPlaneService)
    app = create_app()
    client = app.test_client()

    response = client.get(
        "/operator/control-plane/services",
        headers=_headers(),
    )

    assert response.status_code == 404
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "unknown control-plane section"


def test_control_plane_section_service_error_returns_503(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setenv(
        "AI_TRADING_OPERATOR_TOKEN_MAP",
        json.dumps({"ops@example.com": "ops-token"}),
    )

    class _FakeControlPlaneService:
        def __init__(self, *, service_name: str) -> None:
            assert service_name == "ai-trading"

        def section(self, section: str) -> dict[str, object]:
            assert section == "liveness"
            raise ValueError("liveness unavailable")

    monkeypatch.setattr(app_mod, "ControlPlaneService", _FakeControlPlaneService)
    app = create_app()
    client = app.test_client()

    response = client.get(
        "/operator/control-plane/liveness",
        headers=_headers(),
    )

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "operator control-plane section unavailable"


def test_metrics_endpoint_rejects_non_allowlisted_remote(monkeypatch) -> None:
    monkeypatch.setattr(
        app_mod,
        "_managed_env",
        lambda name, default=None: {
            "AI_TRADING_METRICS_ALLOWED_REMOTE_ADDRS": "127.0.0.1,::1",
        }.get(name, default),
    )
    monkeypatch.setattr(
        app_mod,
        "request",
        SimpleNamespace(remote_addr="203.0.113.10", headers={}),
    )

    response = app_mod._metrics_access_error()

    assert response is not None
    assert response[1] == 403
    assert "metrics access denied" in response[0]
