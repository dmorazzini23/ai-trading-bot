from __future__ import annotations

from ai_trading.app import create_app
from ai_trading.telemetry import runtime_state
from ai_trading.app import create_app
from ai_trading.telemetry import runtime_state


def _build_app(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setattr("ai_trading.config.management.validate_required_env", lambda: None)
    return create_app()


def test_healthz_warming_status(monkeypatch):
    app = _build_app(monkeypatch)
    runtime_state.update_service_status(status="warming_up")
    runtime_state.update_data_provider_state(status="healthy", primary="alpaca", active="alpaca")
    runtime_state.update_broker_status(status="reachable", connected=True)
    client = app.test_client()
    resp = client.get("/healthz")
    data = resp.get_json()
    assert data["status"] == "warming_up"
    assert data["data_provider"]["status"] == "healthy"


def test_healthz_degraded_provider(monkeypatch):
    app = _build_app(monkeypatch)
    runtime_state.update_service_status(status="ready")
    runtime_state.update_data_provider_state(
        status="degraded",
        reason="quota_exhausted",
        http_code=429,
        primary="alpaca",
        active="yahoo",
        using_backup=True,
    )
    runtime_state.update_broker_status(status="reachable", connected=True)
    client = app.test_client()
    resp = client.get("/healthz")
    data = resp.get_json()
    assert data["status"] == "degraded"
    assert data["data_provider"]["http_code"] == 429
    assert data.get("http_code") == 429
    assert data.get("reason") == "quota_exhausted"


def test_healthz_data_status_empty_marks_degraded(monkeypatch):
    app = _build_app(monkeypatch)
    runtime_state.update_service_status(status="ready")
    runtime_state.update_data_provider_state(
        status="healthy",
        primary="alpaca",
        active="alpaca",
        data_status="empty",
        reason="data_source_empty",
    )
    runtime_state.update_broker_status(status="reachable", connected=True)
    client = app.test_client()
    resp = client.get("/healthz")
    data = resp.get_json()
    assert data["status"] == "degraded"
    assert data["data_status"] == "empty"
    assert data.get("reason") in {"data_source_empty", "data_unavailable"}
