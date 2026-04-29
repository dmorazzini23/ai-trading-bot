from __future__ import annotations

from datetime import UTC, datetime

import ai_trading.health_payload as health_payload


def test_runtime_attention_flags_deduplicate_optional_contract_failures() -> None:
    flags = health_payload._build_runtime_attention_flags(
        provider_state={
            "reason": "market_closed",
            "using_backup": True,
            "safe_mode": True,
        },
        broker_state={"open_orders_count": "2", "positions_count": "3"},
        service_state={"status": "halted", "reason": "hard_stop"},
        database_readiness={"configured": True, "ok": False},
        oms_invariants={"enabled": True, "ok": False},
        oms_lifecycle_parity={"enabled": True, "ok": False},
        replay_live_parity_gate={"enabled": True, "ok": False},
        include_optional_contract_failures=True,
    )

    assert flags == [
        "market_closed_non_flat_positions",
        "market_closed_open_orders",
        "provider_backup_active",
        "provider_safe_mode",
        "service_degraded",
        "service_halt_active",
        "replay_live_parity_gate_failed",
        "database_unhealthy",
        "oms_invariants_failed",
        "oms_lifecycle_parity_failed",
    ]


def test_build_runtime_health_payload_marks_required_database_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        health_payload.runtime_state,
        "observe_data_provider_state",
        lambda: {"status": "healthy", "active": "alpaca", "primary": "alpaca"},
    )
    monkeypatch.setattr(
        health_payload.runtime_state,
        "observe_broker_status",
        lambda: {"status": "connected", "connected": True},
    )
    monkeypatch.setattr(
        health_payload.runtime_state,
        "observe_service_status",
        lambda: {"status": "ready", "reason": None},
    )
    monkeypatch.setattr(health_payload.runtime_state, "observe_quote_status", lambda: {})
    monkeypatch.setattr(health_payload, "get_backup_data_provider", lambda: "yahoo")
    monkeypatch.setattr(health_payload, "_model_liveness_snapshot", lambda: {"ok": True})
    monkeypatch.setattr(
        health_payload,
        "_database_readiness_snapshot_cached",
        lambda: {"enabled": True, "configured": True, "ok": False},
    )
    monkeypatch.setattr(
        health_payload,
        "_oms_invariants_snapshot_cached",
        lambda: {"enabled": True, "ok": True},
    )
    monkeypatch.setattr(
        health_payload,
        "_oms_lifecycle_parity_snapshot_cached",
        lambda: {"enabled": True, "ok": True},
    )
    monkeypatch.setattr(
        health_payload,
        "_replay_live_parity_gate_snapshot_cached",
        lambda *, oms_lifecycle_parity=None: {"enabled": True, "ok": True},
    )
    monkeypatch.setattr(
        health_payload,
        "_env_bool",
        lambda name, default: name == "AI_TRADING_HEALTH_REQUIRE_DB_READY",
    )

    payload = health_payload.build_runtime_health_payload(service_name="svc")

    assert payload["service"] == "svc"
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "database_unhealthy"
    assert payload["readiness_failures"] == ["database_unhealthy"]
    assert "database_unhealthy" in payload["attention_flags"]


def test_warmup_market_closed_fast_path_reports_ready(monkeypatch) -> None:
    phase_since = datetime.now(UTC).isoformat()
    monkeypatch.setattr(
        health_payload.runtime_state,
        "observe_data_provider_state",
        lambda: {"status": "healthy", "reason": "market_closed"},
    )
    monkeypatch.setattr(
        health_payload.runtime_state,
        "observe_broker_status",
        lambda: {"status": "connected", "connected": True},
    )
    monkeypatch.setattr(
        health_payload.runtime_state,
        "observe_service_status",
        lambda: {
            "status": "warming",
            "phase": "warmup",
            "reason": "warmup_cycle",
            "phase_since": phase_since,
        },
    )
    monkeypatch.setattr(health_payload.runtime_state, "observe_quote_status", lambda: {})
    monkeypatch.setattr(health_payload, "get_backup_data_provider", lambda: "yahoo")
    monkeypatch.setattr(health_payload, "_model_liveness_snapshot", lambda: {})
    monkeypatch.setattr(
        health_payload,
        "_database_readiness_snapshot_cached",
        lambda: {"enabled": True, "configured": False, "ok": True},
    )
    monkeypatch.setattr(
        health_payload,
        "_oms_invariants_snapshot_cached",
        lambda: {"enabled": False},
    )
    monkeypatch.setattr(
        health_payload,
        "_oms_lifecycle_parity_snapshot_cached",
        lambda: {"enabled": False},
    )
    monkeypatch.setattr(
        health_payload,
        "_replay_live_parity_gate_snapshot_cached",
        lambda *, oms_lifecycle_parity=None: {"enabled": False},
    )
    monkeypatch.setattr(health_payload, "_env_bool", lambda _name, default: bool(default))
    monkeypatch.setattr(health_payload, "_env_float", lambda _name, default: float(default))

    payload = health_payload.build_runtime_health_payload(ok_mode="connectivity")

    assert payload["ok"] is True
    assert payload["status"] == "healthy"
    assert payload["reason"] == "market_closed"
    assert payload["service_state"]["status"] == "ready"
