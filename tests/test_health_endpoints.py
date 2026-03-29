from types import SimpleNamespace
from datetime import UTC, datetime

import pytest

import ai_trading.app as app_module
import ai_trading.health_payload as health_payload_module
from ai_trading.app import create_app
from ai_trading.health import HealthCheck
from ai_trading.telemetry import runtime_state


@pytest.fixture(autouse=True)
def _stub_runtime_state(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "backup": "yahoo",
        "using_backup": False,
        "status": "healthy",
        "consecutive_failures": 0,
        "gap_ratio_recent": 0.0,
        "quote_fresh_ms": 250.0,
        "safe_mode": False,
    }
    broker_state = {
        "status": "reachable",
        "connected": True,
        "latency_ms": 12.5,
        "last_error": None,
    }
    service_state = {"status": "ready"}
    quote_state = {"status": "aligned"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)


def test_app_health_endpoint_shared_port():
    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["service"] == "ai-trading"
    assert payload["timestamp"].endswith("Z")
    assert "provider_state" in payload
    assert payload["provider_state"]["primary"] == "alpaca"
    assert payload["gap_ratio_pct"] == 0.0
    assert payload["quote_fresh_ms"] == 250.0
    assert payload["safe_mode"] is False
    assert "model_liveness" in payload
    assert isinstance(payload["model_liveness"], dict)


def test_standalone_health_server_handler():
    ctx = SimpleNamespace(host="127.0.0.1", port=0, service="ai-trading")
    checker = HealthCheck(ctx=ctx)
    client = checker.app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] in {"healthy", "degraded"}


def test_app_health_unknown_provider_is_not_upgraded_to_healthy(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "backup": None,
        "using_backup": False,
        "status": "unknown",
        "consecutive_failures": 0,
        "last_error_at": None,
        "gap_ratio_recent": None,
        "quote_fresh_ms": None,
        "safe_mode": False,
    }
    broker_state = {
        "status": "reachable",
        "connected": True,
        "latency_ms": None,
        "last_error": None,
    }
    service_state = {"status": "ready"}
    quote_state = {"status": "aligned"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "degraded"
    assert payload["ok"] is False
    assert payload["data_provider"]["status"] == "unknown"


def test_pytest_override_keeps_ok_true(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": None,
        "backup": "yahoo",
        "using_backup": True,
        "status": "down",
        "consecutive_failures": 7,
        "gap_ratio_recent": None,
        "quote_fresh_ms": None,
        "safe_mode": True,
    }
    broker_state = {
        "status": "failed",
        "connected": False,
        "latency_ms": None,
        "last_error": "connection refused",
    }
    service_state = {"status": "ready"}
    quote_state = {"status": "stale"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)

    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["ok"] is True
    assert payload["data_provider"]["status"] == "down"
    assert payload["broker"]["connected"] is False


def test_health_connectivity_mode_requires_known_broker_status(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": False,
        "status": "healthy",
    }
    broker_state = {
        "status": "unknown",
        "connected": None,
    }
    service_state = {"status": "ready"}
    quote_state = {"status": "aligned"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "broker_status_unknown"


def test_health_market_closed_offhours_reports_healthy(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": False,
        "status": "warming_up",
        "data_status": "warming_up",
        "reason": "market_closed",
    }
    broker_state = {
        "status": "connected",
        "connected": True,
        "latency_ms": 15.0,
        "last_error": None,
    }
    service_state = {
        "status": "warming_up",
        "reason": "startup_complete_pending_runtime_health",
        "phase": "active",
    }
    quote_state = {"status": "unknown"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["status"] == "healthy"
    assert payload["reason"] == "market_closed"


def test_health_market_closed_does_not_hide_unknown_broker(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": False,
        "status": "warming_up",
        "data_status": "warming_up",
        "reason": "market_closed",
    }
    broker_state = {
        "status": "unknown",
        "connected": None,
        "latency_ms": None,
        "last_error": None,
    }
    service_state = {
        "status": "warming_up",
        "reason": "startup_complete_pending_runtime_health",
    }
    quote_state = {"status": "unknown"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is False
    assert payload["status"] == "degraded"


def test_health_warmup_market_closed_fast_path_allows_unknown_broker(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": False,
        "status": "warming_up",
        "data_status": "warming_up",
        "reason": "startup_config_resolved",
    }
    broker_state = {
        "status": "unknown",
        "connected": None,
        "latency_ms": None,
        "last_error": None,
    }
    service_state = {
        "status": "warming_up",
        "reason": "warmup_cycle",
        "phase": "warmup",
        "phase_since": datetime.now(UTC).isoformat(),
    }
    quote_state = {"status": "unknown"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(
        health_payload_module,
        "_market_is_closed_now",
        lambda: True,
    )
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["status"] == "healthy"
    assert payload["reason"] == "market_closed"


def test_health_payload_does_not_report_healthy_when_ok_is_false(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": False,
        "status": "healthy",
    }
    broker_state = {
        "status": "degraded",
        "connected": True,
    }
    service_state = {"status": "ready", "reason": "runtime_health_pending"}
    quote_state = {"status": "aligned"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "runtime_health_pending"


def test_pytest_detection_silent_without_hints(monkeypatch, caplog):
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_module, "sys", SimpleNamespace(modules={}))

    caplog.set_level("DEBUG")
    assert app_module._pytest_active() is False
    assert not [record for record in caplog.records if record.message == "PYTEST_DETECT_FALSE"]


def test_health_payload_exposes_provider_reason_code(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "yahoo",
        "backup": "yahoo",
        "using_backup": True,
        "status": "degraded",
        "data_status": "ready",
        "reason": "request_timeout",
        "reason_code": "timeout",
        "reason_detail": "request_timeout",
    }
    broker_state = {
        "status": "connected",
        "connected": True,
        "latency_ms": 10.0,
        "last_error": None,
    }
    service_state = {"status": "ready"}
    quote_state = {"status": "aligned"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    client = app.test_client()
    response = client.get("/healthz")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["data_provider"]["reason_code"] == "timeout"
    assert payload["data_provider"]["reason_detail"] == "request_timeout"
