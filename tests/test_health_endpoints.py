from types import SimpleNamespace
from datetime import UTC, datetime

import pytest

import ai_trading.app as app_module
import ai_trading.health_payload as health_payload_module
from ai_trading.app import create_app
from ai_trading.health_payload import register_health_routes
from ai_trading.health import HealthCheck
from ai_trading.telemetry import runtime_state


def _call_healthz(app):
    if hasattr(app, "test_client"):
        response = app.test_client().get("/healthz")
        return response.get_json(), response.status_code
    handler = app._routes[("/healthz", "GET")]
    result = handler()
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, 200


class _RouteCaptureApp:
    def __init__(self) -> None:
        self._routes = {}

    def route(self, path, *args, **kwargs):
        methods = tuple(str(method).upper() for method in (kwargs.get("methods") or ("GET",)))

        def decorator(func):
            for method in methods:
                self._routes[(path, method)] = func
            return func

        return decorator


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
        "open_orders_count": 0,
        "positions_count": 0,
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
    assert payload["broker"]["open_orders_count"] == 0
    assert payload["broker"]["positions_count"] == 0
    assert payload["attention_flags"] == []


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

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "degraded"
    assert payload["ok"] is False
    assert payload["data_provider"]["status"] == "unknown"


def test_health_surface_exposes_effective_backup_provider_when_runtime_state_missing_backup(
    monkeypatch,
):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "backup": None,
        "using_backup": False,
        "status": "healthy",
        "consecutive_failures": 0,
        "gap_ratio_recent": 0.0,
        "quote_fresh_ms": 100.0,
        "safe_mode": False,
    }
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(health_payload_module, "get_backup_data_provider", lambda: "none")

    app = create_app()
    payload = app.test_client().get("/healthz").get_json()

    assert payload["provider_state"]["backup"] == "none"


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

    assert response.status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "broker_status_unknown"


def test_health_connectivity_mode_rejects_stale_cached_broker_snapshot(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": False,
        "status": "healthy",
    }
    broker_state = {
        "status": "connected",
        "connected": True,
        "fresh": False,
        "open_orders_fresh": False,
        "positions_fresh": True,
        "last_error": "http_500",
        "failed_components": ("open_orders",),
        "consecutive_failures": 2,
        "stale_age_s": 45.0,
        "open_orders_count": 2,
        "positions_count": 1,
    }
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: {"status": "ready"})
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: {"status": "aligned"})
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    response = app.test_client().get("/healthz")
    payload = response.get_json()

    assert response.status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "http_500"
    assert payload["broker"]["connected"] is False
    assert payload["broker"]["fresh"] is False
    assert payload["broker"]["open_orders_count"] == 2
    assert payload["broker"]["positions_count"] == 1
    assert "broker_snapshot_stale" in payload["attention_flags"]
    assert "broker_open_orders_unknown" in payload["attention_flags"]


def test_health_serializes_malformed_failed_components_gracefully(monkeypatch):
    broker_state = {
        "status": "degraded",
        "connected": False,
        "fresh": False,
        "open_orders_fresh": False,
        "positions_fresh": False,
        "failed_components": 42,
    }
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    app = create_app()
    response = app.test_client().get("/healthz")
    payload = response.get_json()

    assert response.status_code == 503
    assert payload["broker"]["failed_components"] == ["42"]


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
        "open_orders_count": 1,
        "positions_count": 7,
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
    assert payload["service_state"]["status"] == "ready"
    assert payload["service_state"]["reason"] == "market_closed"
    assert "market_closed_non_flat_positions" in payload["attention_flags"]
    assert "market_closed_open_orders" in payload["attention_flags"]


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

    assert response.status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"


def test_health_market_closed_does_not_hide_unknown_provider(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": None,
        "using_backup": False,
        "status": "unknown",
        "data_status": "warming_up",
        "reason": "market_closed",
    }
    broker_state = {"status": "connected", "connected": True}
    service_state = {"status": "warming_up", "reason": "startup_complete_pending_runtime_health"}
    quote_state = {"status": "unknown"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(app_module, "_pytest_active", lambda: False)

    response = create_app().test_client().get("/healthz")
    payload = response.get_json()

    assert response.status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "provider_status_unknown"


def test_health_warmup_market_closed_fast_path_requires_known_broker(monkeypatch):
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

    assert response.status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "broker_status_unknown"


def test_health_warmup_market_closed_fast_path_allows_connected_broker(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": False,
        "status": "warming_up",
        "data_status": "warming_up",
        "reason": "startup_config_resolved",
    }
    broker_state = {
        "status": "connected",
        "connected": True,
        "latency_ms": 12.0,
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

    assert response.status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "runtime_health_pending"


def test_health_payload_service_degraded_marks_unhealthy(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": False,
        "status": "healthy",
        "safe_mode": False,
    }
    broker_state = {
        "status": "reachable",
        "connected": True,
    }
    service_state = {
        "status": "degraded",
        "reason": "trade_updates_stream_failed",
        "phase": "active",
        "phase_since": datetime.now(UTC).isoformat(),
    }
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

    assert response.status_code == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "trade_updates_stream_failed"
    assert payload["service_state"]["status"] == "degraded"
    assert "service_degraded" in payload["attention_flags"]
    assert "trade_updates_stream_degraded" in payload["attention_flags"]


def test_control_plane_snapshot_surfaces_runtime_attention_flags(monkeypatch):
    provider_state = {
        "primary": "alpaca",
        "active": "alpaca",
        "using_backup": True,
        "status": "degraded",
        "reason": "provider_fallback_active",
        "safe_mode": True,
    }
    broker_state = {
        "status": "reachable",
        "connected": True,
        "open_orders_count": 1,
        "positions_count": 2,
    }
    service_state = {
        "status": "degraded",
        "reason": "trade_updates_stream_failed",
        "phase": "active",
    }
    quote_state = {"status": "aligned"}
    monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: provider_state)
    monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: broker_state)
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: service_state)
    monkeypatch.setattr(runtime_state, "observe_quote_status", lambda: quote_state)
    monkeypatch.setattr(health_payload_module, "_model_liveness_snapshot", lambda: {})
    monkeypatch.setattr(
        health_payload_module,
        "_database_readiness_snapshot",
        lambda: {"enabled": True, "configured": True, "ok": False},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_oms_invariants_snapshot",
        lambda: {"enabled": True, "ok": False},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_oms_lifecycle_parity_snapshot",
        lambda: {"enabled": True, "ok": False},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {"enabled": True, "ok": False},
    )
    monkeypatch.setattr(health_payload_module, "_runtime_performance_snapshot", lambda: {})
    monkeypatch.setattr(health_payload_module, "_manual_override_snapshot", lambda: {})
    monkeypatch.setattr(health_payload_module, "_governance_snapshot", lambda: {})

    snapshot = health_payload_module.build_control_plane_snapshot()

    assert snapshot["service_state"]["reason"] == "trade_updates_stream_failed"
    assert "provider_backup_active" in snapshot["attention_flags"]
    assert "provider_safe_mode" in snapshot["attention_flags"]
    assert "service_degraded" in snapshot["attention_flags"]
    assert "trade_updates_stream_degraded" in snapshot["attention_flags"]
    assert "database_unhealthy" in snapshot["attention_flags"]
    assert "oms_invariants_failed" in snapshot["attention_flags"]
    assert "oms_lifecycle_parity_failed" in snapshot["attention_flags"]
    assert "replay_live_parity_gate_failed" in snapshot["attention_flags"]


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


def test_health_route_builder_failure_returns_503_json():
    app = _RouteCaptureApp()

    def broken_builder():
        raise RuntimeError("builder down")

    register_health_routes(
        app,
        payload_builder=broken_builder,
        response_builder=lambda payload, status: (dict(payload), status),
    )

    payload, status = _call_healthz(app)

    assert status == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["error"] == "builder down"


def test_health_route_false_ok_returns_503():
    app = _RouteCaptureApp()

    register_health_routes(
        app,
        payload_builder=lambda: {"ok": False, "status": "degraded", "reason": "not_ready"},
        response_builder=lambda payload, status: (payload, status),
    )

    payload, status = _call_healthz(app)

    assert status == 503
    assert payload["ok"] is False
    assert payload["reason"] == "not_ready"


def test_health_route_response_failure_returns_503_json_tuple():
    app = _RouteCaptureApp()
    calls = {"count": 0}

    def flaky_response(payload, status):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("response down")
        return dict(payload), status

    register_health_routes(
        app,
        payload_builder=lambda: {"ok": True, "status": "healthy"},
        response_builder=flaky_response,
    )

    payload, status = _call_healthz(app)

    assert status == 503
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["error"] == "response down"


def test_runtime_health_payload_includes_database_snapshot(monkeypatch):
    monkeypatch.setattr(
        health_payload_module,
        "_database_readiness_snapshot",
        lambda: {
            "enabled": True,
            "configured": True,
            "ok": True,
            "connected": True,
            "backend": "sqlite",
        },
    )
    payload = health_payload_module.build_runtime_health_payload()
    assert "database" in payload
    assert payload["database"]["ok"] is True
    assert payload["database"]["backend"] == "sqlite"


def test_runtime_health_payload_db_requirement_marks_degraded(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_DB_READY", "1")
    monkeypatch.setattr(
        health_payload_module,
        "_database_readiness_snapshot",
        lambda: {
            "enabled": True,
            "configured": True,
            "ok": False,
            "connected": False,
            "backend": "postgres",
            "error": "connection refused",
        },
    )
    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload.get("reason") == "database_unhealthy"


@pytest.mark.parametrize("market_closed", [False, True])
def test_runtime_health_reports_strategy_state_without_model_readiness(
    monkeypatch, market_closed,
):
    if market_closed:
        monkeypatch.setattr(runtime_state, "observe_data_provider_state", lambda: {
            "status": "idle", "reason": "market_closed", "market_open": False,
        })
        monkeypatch.setattr(runtime_state, "observe_broker_status", lambda: {
            "status": "connected", "connected": True,
            "positions_count": 0, "open_orders_count": 0,
        })
        monkeypatch.setattr(runtime_state, "observe_service_status", lambda: {
            "status": "ready", "phase": "active", "reason": "market_closed",
        })
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    monkeypatch.setattr(
        health_payload_module,
        "_operating_mode_snapshot",
        lambda: {
            "strategy_state": "no_qualified_strategy",
            "qualified_strategy": False,
            "mode": "paper_rule_sampling",
            "rule_paper_sampling_enabled": True,
            "results_are_strategy_performance": False,
            "promotion_authority": False,
        },
    )
    monkeypatch.setattr(
        health_payload_module,
        "_latest_training_attempt_snapshot",
        lambda: {
            "available": True,
            "status": "skipped",
            "reason": "no_qualified_candidate",
            "generated_at": "2026-08-13T21:00:00Z",
            "runtime_authority": False,
            "promotion_authority": False,
            "live_money_authority": False,
        },
    )

    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )

    assert payload["ok"] is True
    assert payload["status"] != "degraded"
    assert payload["readiness_failures"] == []
    assert "day_sleeve_model" not in payload
    assert "day_sleeve_model" not in payload["readiness_gates"]
    assert "required_model_stale" not in payload["readiness_failures"]
    assert "required_model_stale" not in payload["attention_flags"]
    assert payload["operating_mode"]["mode"] == "paper_rule_sampling"
    assert payload["operating_mode"]["strategy_state"] == "no_qualified_strategy"
    assert payload["latest_training_attempt"]["status"] == "skipped"
    assert payload["latest_training_attempt"]["reason"] == "no_qualified_candidate"
    assert payload["latest_training_attempt"]["runtime_authority"] is False
    assert payload["entry_control"]["paper_evidence_allowed"] is (not market_closed)


def test_runtime_health_payload_db_requirement_rejects_unconfigured_db(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_DB_READY", "1")
    monkeypatch.setattr(
        health_payload_module,
        "_database_readiness_snapshot",
        lambda: {
            "enabled": True,
            "configured": False,
            "ok": False,
            "reason": "database_not_configured",
        },
    )

    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )

    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload.get("reason") == "database_unhealthy"


def test_runtime_health_payload_includes_oms_invariants_snapshot(monkeypatch):
    monkeypatch.setattr(
        health_payload_module,
        "_oms_invariants_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": True,
            "scanned_intents": 12,
            "total_violations": 0,
        },
    )
    payload = health_payload_module.build_runtime_health_payload()
    assert "oms_invariants" in payload
    assert payload["oms_invariants"]["ok"] is True
    assert payload["oms_invariants"]["scanned_intents"] == 12


def test_runtime_health_payload_oms_invariants_requirement_marks_degraded(monkeypatch):
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS", "1")
    monkeypatch.setattr(
        health_payload_module,
        "_oms_invariants_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": False,
            "total_violations": 3,
        },
    )
    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload.get("reason") == "oms_invariants_failed"
    assert "oms_invariants_failed" in payload["readiness_failures"]
    assert "oms_invariants_failed" in payload["attention_flags"]
    assert payload["readiness_gates"]["oms_invariants"]["status"] == "required_failed"
    assert payload["readiness_gates"]["oms_invariants"]["required"] is True
    assert "repair" in payload["readiness_gates"]["oms_invariants"]["action"]


def test_runtime_health_payload_oms_observe_only_failures_do_not_fail_readiness(
    monkeypatch,
):
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS", "0")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY", "0")
    monkeypatch.setattr(
        health_payload_module,
        "_oms_invariants_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": False,
            "reason": "missing_terminal_event",
            "total_violations": 3,
        },
    )
    monkeypatch.setattr(
        health_payload_module,
        "_oms_lifecycle_parity_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": False,
            "reason": "missing_submit_ack",
            "total_violations": 2,
        },
    )
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {"enabled": False, "available": False, "ok": True},
    )

    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )

    assert payload["ok"] is True
    assert payload["status"] == "healthy"
    assert payload["readiness_failures"] == []
    assert "oms_invariants_failed" in payload["attention_flags"]
    assert "oms_lifecycle_parity_failed" in payload["attention_flags"]
    assert payload["readiness_gates"]["oms_invariants"]["status"] == "observed_failure"
    assert payload["readiness_gates"]["oms_invariants"]["required"] is False
    assert payload["readiness_gates"]["oms_invariants"]["detail"] == "missing_terminal_event"
    assert payload["readiness_gates"]["oms_lifecycle_parity"]["status"] == "observed_failure"
    assert payload["readiness_gates"]["oms_lifecycle_parity"]["required"] is False


def test_runtime_health_payload_includes_oms_lifecycle_parity_snapshot(monkeypatch):
    monkeypatch.setattr(
        health_payload_module,
        "_oms_lifecycle_parity_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": True,
            "scanned_intents": 10,
            "total_violations": 0,
        },
    )
    payload = health_payload_module.build_runtime_health_payload()
    assert "oms_lifecycle_parity" in payload
    assert payload["oms_lifecycle_parity"]["ok"] is True
    assert payload["oms_lifecycle_parity"]["scanned_intents"] == 10


def test_runtime_health_payload_oms_lifecycle_parity_requirement_marks_degraded(
    monkeypatch,
):
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY", "1")
    monkeypatch.setattr(
        health_payload_module,
        "_oms_lifecycle_parity_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": False,
            "total_violations": 2,
        },
    )
    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload.get("reason") == "oms_lifecycle_parity_failed"
    assert "oms_lifecycle_parity_failed" in payload["readiness_failures"]
    assert "oms_lifecycle_parity_failed" in payload["attention_flags"]
    assert (
        payload["readiness_gates"]["oms_lifecycle_parity"]["status"]
        == "required_failed"
    )
    assert payload["readiness_gates"]["oms_lifecycle_parity"]["required"] is True


def test_runtime_health_payload_requires_oms_invariants_by_default_outside_pytest(
    monkeypatch,
):
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(health_payload_module, "_health_snapshot_cache_enabled", lambda: False)
    monkeypatch.setattr(
        health_payload_module,
        "_oms_invariants_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": False,
            "total_violations": 2,
        },
    )
    monkeypatch.setattr(
        health_payload_module,
        "_oms_lifecycle_parity_snapshot",
        lambda: {"enabled": False},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {"enabled": False, "available": False, "ok": True},
    )
    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )
    assert payload["ok"] is False
    assert payload.get("reason") == "oms_invariants_failed"


def test_runtime_health_payload_requires_oms_lifecycle_parity_by_default_outside_pytest(
    monkeypatch,
):
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(health_payload_module, "_health_snapshot_cache_enabled", lambda: False)
    monkeypatch.setattr(
        health_payload_module,
        "_oms_invariants_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": True,
            "total_violations": 0,
        },
    )
    monkeypatch.setattr(
        health_payload_module,
        "_oms_lifecycle_parity_snapshot",
        lambda: {
            "enabled": True,
            "available": True,
            "ok": False,
            "total_violations": 1,
        },
    )
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {"enabled": False, "available": False, "ok": True},
    )
    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )
    assert payload["ok"] is False
    assert payload.get("reason") == "oms_lifecycle_parity_failed"


def test_runtime_health_payload_includes_replay_live_parity_gate(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REPLAY_GATE_ENFORCE_IN_PAPER", "0")
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {
            "enabled": True,
            "available": True,
            "ok": True,
            "status": "pass",
            "reason": "ok",
        },
    )
    payload = health_payload_module.build_runtime_health_payload()
    assert "replay_live_parity_gate" in payload
    assert payload["replay_live_parity_gate"]["ok"] is True
    assert payload["replay_live_parity_gate"]["status"] == "pass"
    assert payload["entry_control"]["stage"] == "paper"
    assert payload["entry_control"]["replay_gate_enforced"] is False
    assert payload["entry_control"]["paper_evidence_allowed"] is True


def test_cached_background_snapshot_returns_placeholder_then_cached(monkeypatch):
    health_payload_module._HEALTH_SNAPSHOT_CACHE.clear()

    class _ImmediateThread:
        def __init__(self, *, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(health_payload_module, "Thread", _ImmediateThread)

    first = health_payload_module._cached_background_snapshot(
        name="demo_snapshot",
        ttl_seconds=30.0,
        placeholder={"enabled": True, "ok": False},
        builder=lambda: {"enabled": True, "ok": True, "value": 7},
    )
    second = health_payload_module._cached_background_snapshot(
        name="demo_snapshot",
        ttl_seconds=30.0,
        placeholder={"enabled": True, "ok": False},
        builder=lambda: {"enabled": True, "ok": True, "value": 7},
    )

    assert first["enabled"] is True
    assert first["ok"] is False
    assert first["refreshing"] is True
    assert second["ok"] is True
    assert second["value"] == 7
    assert second["refreshing"] is False


def test_expensive_health_snapshots_use_slow_refresh_default(monkeypatch):
    captured: dict[str, float] = {}

    def _cached_stub(*, name, ttl_seconds, placeholder, builder):
        captured[name] = ttl_seconds
        return dict(placeholder)

    monkeypatch.setattr(health_payload_module, "_health_snapshot_cache_enabled", lambda: True)
    monkeypatch.setattr(health_payload_module, "_env_float", lambda _name, default: default)
    monkeypatch.setattr(health_payload_module, "_cached_background_snapshot", _cached_stub)

    health_payload_module._oms_invariants_snapshot_cached()
    health_payload_module._oms_lifecycle_parity_snapshot_cached()
    health_payload_module._replay_live_parity_gate_snapshot_cached()

    assert captured["oms_invariants"] == 300.0
    assert captured["oms_lifecycle_parity"] == 300.0
    assert captured["replay_live_parity_gate"] == 300.0


def test_required_health_snapshots_bypass_async_placeholders(monkeypatch):
    monkeypatch.setattr(health_payload_module, "_health_snapshot_cache_enabled", lambda: True)
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_DB_READY", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_OMS_INVARIANTS", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")

    def _unexpected_cache(**_kwargs):
        raise AssertionError("required health gates should sample synchronously")

    monkeypatch.setattr(health_payload_module, "_cached_background_snapshot", _unexpected_cache)
    monkeypatch.setattr(
        health_payload_module,
        "_database_readiness_snapshot",
        lambda: {"enabled": True, "ok": True, "sample": "db"},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_oms_invariants_snapshot",
        lambda: {"enabled": True, "ok": True, "sample": "oms"},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_oms_lifecycle_parity_snapshot",
        lambda: {"enabled": True, "ok": True, "sample": "lifecycle"},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {"enabled": True, "ok": True, "sample": "replay"},
    )

    assert health_payload_module._database_readiness_snapshot_cached()["sample"] == "db"
    assert health_payload_module._oms_invariants_snapshot_cached()["sample"] == "oms"
    assert (
        health_payload_module._oms_lifecycle_parity_snapshot_cached()["sample"]
        == "lifecycle"
    )
    assert (
        health_payload_module._replay_live_parity_gate_snapshot_cached()["sample"]
        == "replay"
    )


def test_replay_live_parity_gate_does_not_cache_warming_oms_snapshot(monkeypatch):
    monkeypatch.setattr(health_payload_module, "_health_snapshot_cache_enabled", lambda: True)
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {
            "enabled": True,
            "available": False,
            "ok": False,
            "reason": "oms_lifecycle_parity_available",
        },
    )

    def _unexpected_cache(**_kwargs):
        raise AssertionError("warming OMS parity should not seed replay gate cache")

    monkeypatch.setattr(health_payload_module, "_cached_background_snapshot", _unexpected_cache)

    payload = health_payload_module._replay_live_parity_gate_snapshot_cached(
        oms_lifecycle_parity={
            "enabled": True,
            "available": False,
            "ok": False,
            "refreshing": True,
            "reason": "warming_up",
        },
    )

    assert payload["available"] is False
    assert payload["reason"] == "oms_lifecycle_parity_available"


def test_runtime_health_payload_replay_live_parity_requirement_marks_degraded(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {
            "enabled": True,
            "available": True,
            "ok": False,
            "status": "fail",
            "reason": "replay_counterfactual",
        },
    )
    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload.get("reason") == "replay_live_parity_gate_failed"
    assert payload["entry_control"]["stage"] == "shadow"
    assert payload["entry_control"]["live_new_exposure_allowed"] is False
    assert payload["entry_control"]["risk_reducing_orders_allowed"] is True


@pytest.mark.parametrize(
    "gate_snapshot",
    [
        {
            "enabled": False,
            "available": False,
            "ok": True,
            "status": "disabled",
            "reason": "disabled",
        },
        {
            "enabled": True,
            "available": False,
            "ok": True,
            "status": "unavailable",
            "reason": "artifact_unavailable",
        },
    ],
)
def test_runtime_health_required_replay_gate_fails_closed_when_not_available(
    monkeypatch,
    gate_snapshot,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: gate_snapshot,
    )

    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )

    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "replay_live_parity_gate_failed"
    assert payload["readiness_failures"] == ["replay_live_parity_gate_failed"]
    assert "replay_live_parity_gate_failed" in payload["attention_flags"]
    assert payload["entry_control"]["stage"] == "shadow"
    assert payload["entry_control"]["live_new_exposure_allowed"] is False
    assert payload["entry_control"]["risk_reducing_orders_allowed"] is True


def test_runtime_health_replay_failure_is_monitor_only_for_paper_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REPLAY_GATE_ENFORCE_IN_PAPER", "0")
    monkeypatch.setattr(
        runtime_state,
        "observe_service_status",
        lambda: {
            "status": "degraded",
            "reason": "replay_live_parity_gate_failed",
        },
    )
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {
            "enabled": True,
            "available": True,
            "ok": False,
            "status": "fail",
            "reason": "replay_counterfactual",
            "failed_checks": ["candidate_positive_net_edge"],
        },
    )

    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )

    assert payload["ok"] is True
    assert payload["status"] == "healthy"
    assert "replay_live_parity_gate_failed" in payload["attention_flags"]
    assert "replay_live_parity_gate_failed" not in payload["readiness_failures"]
    assert payload["entry_control"] == {
        "execution_mode": "paper",
        "stage": "shadow",
        "replay_gate_ok": False,
        "replay_gate_required": True,
        "replay_gate_enforced": False,
        "paper_evidence_allowed": True,
        "live_new_exposure_allowed": False,
        "risk_reducing_orders_allowed": True,
        "reason": "replay_counterfactual",
        "failed_checks": ["candidate_positive_net_edge"],
    }

    app = _RouteCaptureApp()
    register_health_routes(
        app,
        payload_builder=lambda: payload,
        response_builder=lambda response_payload, status: (response_payload, status),
    )
    _response_payload, status = _call_healthz(app)
    assert status == 200


def test_runtime_health_replay_failure_can_be_enforced_in_paper(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REPLAY_GATE_ENFORCE_IN_PAPER", "1")
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {
            "enabled": True,
            "available": True,
            "ok": False,
            "status": "fail",
            "reason": "replay_counterfactual",
            "failed_checks": ["candidate_positive_net_edge"],
        },
    )

    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )

    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["readiness_failures"] == ["replay_live_parity_gate_failed"]
    assert payload["entry_control"]["stage"] == "shadow"
    assert payload["entry_control"]["replay_gate_enforced"] is True
    assert payload["entry_control"]["paper_evidence_allowed"] is False

    app = _RouteCaptureApp()
    register_health_routes(
        app,
        payload_builder=lambda: payload,
        response_builder=lambda response_payload, status: (response_payload, status),
    )
    _response_payload, status = _call_healthz(app)
    assert status == 503


_PARITY_CHECK_NAMES = (
    "replay_available",
    "replay_contract",
    "replay_fresh",
    "replay_source_fresh",
    "replay_violations",
    "replay_counterfactual",
    "oms_lifecycle_parity_available",
    "oms_lifecycle_parity_consistent",
)


def _replay_gate_with_failures(*failed: str) -> dict:
    checks = {name: name not in failed for name in _PARITY_CHECK_NAMES}
    failed_checks = [name for name in _PARITY_CHECK_NAMES if name in failed]
    return {
        "enabled": True,
        "available": True,
        "ok": not failed_checks,
        "status": "fail" if failed_checks else "pass",
        "reason": failed_checks[0] if failed_checks else "ok",
        "failed_checks": failed_checks,
        "checks": checks,
        "thresholds": {"require_counterfactual_passed": True},
        "observed": {
            "replay_counterfactual_passed": "replay_counterfactual" not in failed,
            "replay_counterfactual_available": True,
            "replay_counterfactual_candidate_samples": 92,
            "replay_counterfactual_required_samples": 250,
            "replay_counterfactual_candidate_net_edge_bps": -5.69594400285975,
            "replay_counterfactual_required_net_edge_bps": 0.0,
        },
        "replay_governance": {
            "counterfactual": {
                "passed": "replay_counterfactual" not in failed,
                "checks": {
                    "candidate_net_edge_positive": False,
                    "drawdown_non_regression": True,
                    "net_edge_non_regression": True,
                    "sample_size": False,
                },
            }
        },
    }


def _build_replay_scoped_payload(monkeypatch, *, mode: str, gate: dict, service_state: dict):
    if mode == "live":
        monkeypatch.setenv("EXECUTION_MODE", "live")
        monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    else:
        monkeypatch.setenv("EXECUTION_MODE", "paper")
        monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")
    monkeypatch.setenv("AI_TRADING_HEALTH_REPLAY_GATE_ENFORCE_IN_PAPER", "0")
    monkeypatch.setattr(runtime_state, "observe_service_status", lambda: dict(service_state))
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: gate,
    )
    return health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )


_REPLAY_SERVICE_STATES = [
    {"status": "ready", "reason": "market_closed"},
    {"status": "degraded", "reason": "replay_live_parity_gate_failed"},
]


@pytest.mark.parametrize("service_state", _REPLAY_SERVICE_STATES)
def test_paper_counterfactual_only_failure_withholds_parity_flag(
    monkeypatch, service_state
) -> None:
    gate = _replay_gate_with_failures("replay_counterfactual")
    payload = _build_replay_scoped_payload(
        monkeypatch, mode="paper", gate=gate, service_state=service_state
    )

    assert "replay_live_parity_gate_failed" not in payload["attention_flags"]
    # Everything that drives decisions is exactly what it was before.
    assert payload["ok"] is True
    assert payload["status"] == "healthy"
    assert payload["readiness_failures"] == []
    assert payload["entry_control"] == {
        "execution_mode": "paper",
        "stage": "shadow",
        "replay_gate_ok": False,
        "replay_gate_required": True,
        "replay_gate_enforced": False,
        "paper_evidence_allowed": True,
        "live_new_exposure_allowed": False,
        "risk_reducing_orders_allowed": True,
        "reason": "replay_counterfactual",
        "failed_checks": ["replay_counterfactual"],
    }
    block = payload["replay_live_parity_gate"]
    assert block["ok"] is False
    assert block["failed_checks"] == ["replay_counterfactual"]
    assert {key: block[key] for key in gate} == gate
    assert block["parity_ok"] is True
    assert block["parity_failed_checks"] == []
    assert block["replay_economics"] == {
        "check": "replay_counterfactual",
        "required": True,
        "available": True,
        "qualified": False,
        "sample_size": 92,
        "min_samples": 250,
        "net_edge_bps": -5.69594400285975,
        "min_net_edge_bps": 0.0,
        "failed_criteria": ["candidate_net_edge_positive", "sample_size"],
        "reason": "insufficient_samples",
        "withheld_attention_flag": "replay_live_parity_gate_failed",
        "summary": (
            "Replay economics not qualified: 92 of 250 capped samples, net edge "
            "-5.70 bps (minimum 0.00 bps). Paper mode: informational, not a "
            "parity failure, so the replay parity attention flag is not raised."
        ),
    }
    # The cached gate snapshot itself is not mutated.
    assert "replay_economics" not in gate


@pytest.mark.parametrize("service_state", _REPLAY_SERVICE_STATES)
@pytest.mark.parametrize(
    "failed",
    [
        ("oms_lifecycle_parity_consistent",),
        ("replay_fresh",),
        ("replay_fresh", "replay_counterfactual"),
        ("oms_lifecycle_parity_consistent", "replay_counterfactual"),
    ],
)
def test_paper_real_parity_failure_keeps_parity_flag(
    monkeypatch, failed, service_state
) -> None:
    gate = _replay_gate_with_failures(*failed)
    payload = _build_replay_scoped_payload(
        monkeypatch, mode="paper", gate=gate, service_state=service_state
    )

    assert "replay_live_parity_gate_failed" in payload["attention_flags"]
    assert payload["ok"] is True
    assert payload["readiness_failures"] == []
    assert payload["entry_control"]["stage"] == "shadow"
    assert payload["entry_control"]["paper_evidence_allowed"] is True
    assert payload["entry_control"]["failed_checks"] == list(gate["failed_checks"])
    block = payload["replay_live_parity_gate"]
    assert block["ok"] is False
    assert block["failed_checks"] == gate["failed_checks"]
    assert block["parity_ok"] is False
    assert block["parity_failed_checks"] == [
        name for name in gate["failed_checks"] if name != "replay_counterfactual"
    ]
    assert block["replay_economics"]["withheld_attention_flag"] is None
    assert block["replay_economics"]["qualified"] is (
        "replay_counterfactual" not in failed
    )


def test_paper_unavailable_replay_gate_keeps_parity_flag(monkeypatch) -> None:
    gate = {"enabled": True, "available": False, "ok": False, "error": "boom"}
    payload = _build_replay_scoped_payload(
        monkeypatch,
        mode="paper",
        gate=gate,
        service_state={"status": "ready", "reason": "market_closed"},
    )

    assert "replay_live_parity_gate_failed" in payload["attention_flags"]
    assert payload["replay_live_parity_gate"]["parity_ok"] is False
    assert payload["replay_live_parity_gate"]["replay_economics"]["reason"] == (
        "counterfactual_unavailable"
    )


def test_paper_enforced_counterfactual_only_failure_keeps_parity_flag(monkeypatch) -> None:
    gate = _replay_gate_with_failures("replay_counterfactual")
    monkeypatch.setenv("AI_TRADING_HEALTH_REPLAY_GATE_ENFORCE_IN_PAPER", "1")
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    monkeypatch.setenv("AI_TRADING_HEALTH_REQUIRE_REPLAY_LIVE_PARITY_GATE", "1")
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: gate,
    )
    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )

    assert payload["readiness_failures"] == ["replay_live_parity_gate_failed"]
    assert "replay_live_parity_gate_failed" in payload["attention_flags"]
    assert payload["replay_live_parity_gate"]["replay_economics"][
        "withheld_attention_flag"
    ] is None


@pytest.mark.parametrize("service_state", _REPLAY_SERVICE_STATES)
def test_live_counterfactual_only_failure_is_unchanged(monkeypatch, service_state) -> None:
    gate = _replay_gate_with_failures("replay_counterfactual")
    payload = _build_replay_scoped_payload(
        monkeypatch, mode="live", gate=gate, service_state=service_state
    )

    assert "replay_live_parity_gate_failed" in payload["attention_flags"]
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "replay_live_parity_gate_failed"
    assert payload["readiness_failures"] == ["replay_live_parity_gate_failed"]
    assert payload["entry_control"]["stage"] == "shadow"
    assert payload["entry_control"]["replay_gate_enforced"] is True
    assert payload["entry_control"]["live_new_exposure_allowed"] is False
    assert payload["entry_control"]["failed_checks"] == ["replay_counterfactual"]
    # The live gate block is passed through untouched.
    assert payload["replay_live_parity_gate"] == gate
    assert "replay_economics" not in payload["replay_live_parity_gate"]


@pytest.mark.parametrize(
    "gate",
    [
        {},
        {"observed": None, "thresholds": "x", "replay_governance": []},
        {
            "observed": {
                "replay_counterfactual_candidate_samples": object(),
                "replay_counterfactual_required_samples": float("inf"),
                "replay_counterfactual_candidate_net_edge_bps": float("inf"),
                "replay_counterfactual_required_net_edge_bps": "abc",
            },
            "replay_governance": {"counterfactual": {"checks": None}},
        },
    ],
)
def test_replay_economics_summary_never_raises_on_malformed_gate(gate) -> None:
    summary = health_payload_module._replay_economics_summary(
        gate, withheld_attention_flag=None
    )

    assert summary["qualified"] is False
    assert summary["net_edge_bps"] is None
    assert summary["reason"] == "counterfactual_unavailable"
    assert summary["withheld_attention_flag"] is None


def test_runtime_health_payload_requires_replay_live_parity_by_default_outside_pytest(
    monkeypatch,
) -> None:
    def _fake_get_env(key, default=None, cast=None, **_kwargs):
        if key == "PYTEST_CURRENT_TEST":
            return ""
        if key == "PYTEST_RUNNING":
            return False
        if key == "EXECUTION_MODE":
            return "live"
        return default

    monkeypatch.setattr(health_payload_module, "get_env", _fake_get_env)
    monkeypatch.setattr(health_payload_module, "_health_snapshot_cache_enabled", lambda: False)
    monkeypatch.setattr(
        health_payload_module,
        "_oms_invariants_snapshot",
        lambda **_kwargs: {"enabled": True, "available": True, "ok": True},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_oms_lifecycle_parity_snapshot",
        lambda **_kwargs: {"enabled": True, "available": True, "ok": True},
    )
    monkeypatch.setattr(
        health_payload_module,
        "_replay_live_parity_gate_snapshot",
        lambda **_kwargs: {
            "enabled": True,
            "available": True,
            "ok": False,
            "status": "fail",
            "reason": "replay_counterfactual",
        },
    )
    payload = health_payload_module.build_runtime_health_payload(
        force_ok_for_pytest=False,
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload.get("reason") == "replay_live_parity_gate_failed"


def _stalled_decision_loop() -> dict:
    return {
        "monitored": True,
        "stalled": True,
        "alert_only": True,
        "seconds_since_progress": 1200.0,
        "stall_threshold_seconds": 900.0,
    }


@pytest.mark.parametrize(
    "builder_kwargs",
    [
        {},
        {"force_ok_for_pytest": False, "healthy_status_mode": "healthy", "ok_mode": "connectivity"},
    ],
)
def test_runtime_health_decision_loop_stall_is_non_fatal_flag(monkeypatch, builder_kwargs):
    monkeypatch.setattr(
        health_payload_module,
        "_decision_loop_snapshot",
        lambda: {"monitored": True, "stalled": False, "alert_only": True},
    )
    baseline = health_payload_module.build_runtime_health_payload(**builder_kwargs)
    monkeypatch.setattr(health_payload_module, "_decision_loop_snapshot", _stalled_decision_loop)
    stalled = health_payload_module.build_runtime_health_payload(**builder_kwargs)

    assert "decision_loop_stalled" not in baseline["attention_flags"]
    assert "decision_loop_stalled" in stalled["attention_flags"]
    assert stalled["decision_loop"]["stalled"] is True
    for key in ("ok", "status", "readiness_failures", "entry_control"):
        assert stalled[key] == baseline[key], key


def test_control_plane_snapshot_reports_decision_loop_stall(monkeypatch):
    monkeypatch.setattr(health_payload_module, "_decision_loop_snapshot", _stalled_decision_loop)

    snapshot = health_payload_module.build_control_plane_snapshot()

    assert "decision_loop_stalled" in snapshot["attention_flags"]
    assert snapshot["decision_loop"]["stalled"] is True


def test_decision_loop_snapshot_never_raises(monkeypatch):
    from ai_trading.monitoring import decision_loop_watchdog

    def _boom(**_kwargs):
        raise RuntimeError("watchdog state unavailable")

    monkeypatch.setattr(decision_loop_watchdog, "get_decision_loop_snapshot", _boom)

    snapshot = health_payload_module._decision_loop_snapshot()
    assert snapshot["stalled"] is False
    assert snapshot["monitored"] is False
    payload = health_payload_module.build_runtime_health_payload()
    assert "decision_loop_stalled" not in payload["attention_flags"]
    assert payload["decision_loop"]["error"] == "watchdog state unavailable"
