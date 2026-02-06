from types import SimpleNamespace

import pytest

import ai_trading.app as app_module
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


def test_standalone_health_server_handler():
    ctx = SimpleNamespace(host="127.0.0.1", port=0, service="ai-trading")
    checker = HealthCheck(ctx=ctx)
    client = checker.app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] in {"healthy", "degraded"}


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


def test_pytest_detection_silent_without_hints(monkeypatch, caplog):
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(app_module, "sys", SimpleNamespace(modules={}))

    caplog.set_level("DEBUG")
    assert app_module._pytest_active() is False
    assert not [record for record in caplog.records if record.message == "PYTEST_DETECT_FALSE"]
