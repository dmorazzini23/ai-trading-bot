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


def test_runtime_health_payload_replay_live_parity_requirement_marks_degraded(
    monkeypatch,
) -> None:
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


def test_runtime_health_payload_requires_replay_live_parity_by_default_outside_pytest(
    monkeypatch,
) -> None:
    def _fake_get_env(key, default=None, cast=None, **_kwargs):
        if key == "PYTEST_CURRENT_TEST":
            return ""
        if key == "PYTEST_RUNNING":
            return False
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
