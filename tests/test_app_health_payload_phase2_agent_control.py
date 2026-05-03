from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

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


def test_execution_quality_governor_snapshot_reads_compact_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "execution_quality_governor_latest.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-01T15:00:00Z",
                "status": {"gate_passed": True, "mode": "derisk"},
                "observed": {"p90_spread_bps": 9.5},
                "actions": {"blocked_count": 1},
                "window": {"minutes": 60, "event_count": 10},
                "by_symbol": [{"symbol": "MSFT", "p90_spread_bps": 55.0}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "AI_TRADING_EXECUTION_QUALITY_GOVERNOR_REPORT_PATH",
        str(artifact),
    )

    snapshot = health_payload._execution_quality_governor_snapshot()

    assert snapshot["available"] is True
    assert snapshot["path"] == str(artifact)
    assert snapshot["status"]["mode"] == "derisk"
    assert snapshot["observed"]["p90_spread_bps"] == 9.5
    assert snapshot["actions"]["blocked_count"] == 1
    assert snapshot["by_symbol"][0]["symbol"] == "MSFT"


def test_live_cost_model_snapshot_reads_symbol_session_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "live_cost_model_latest.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-01T15:00:00Z",
                "status": {"available": True, "status": "ready", "mode": "observe"},
                "observed": {"p90_total_cost_bps": 8.5},
                "window": {"minutes": 390, "sample_count": 12},
                "sources": {
                    "execution_quality_events": {"rows_read": 12, "rows_used": 12}
                },
                "alerts": {"cost_threshold_breaches": []},
                "by_symbol_side_session": [
                    {
                        "symbol": "AAPL",
                        "side": "buy",
                        "session_regime": "midday",
                        "mean_total_cost_bps": 4.0,
                        "last_observed_at": "2026-05-01T14:58:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_LIVE_COST_MODEL_PATH", str(artifact))

    snapshot = health_payload._live_cost_model_snapshot()

    assert snapshot["available"] is True
    assert snapshot["path"] == str(artifact)
    assert snapshot["status"]["status"] == "ready"
    assert snapshot["observed"]["p90_total_cost_bps"] == 8.5
    assert snapshot["window"]["sample_count"] == 12
    assert snapshot["sources"]["execution_quality_events"]["rows_used"] == 12
    assert snapshot["alerts"]["cost_threshold_breaches"] == []
    assert snapshot["last_observed_at"] == "2026-05-01T14:58:00Z"
    assert snapshot["last_sample_age_seconds"] is not None
    assert snapshot["by_symbol_side_session"][0]["symbol"] == "AAPL"


def test_symbol_universe_scorecard_snapshot_reads_policy_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "symbol_universe_scorecard_latest.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at": "2026-05-01T15:00:00Z",
                "status": {"available": True, "status": "ready", "mode": "observe"},
                "summary": {"symbol_count": 2, "disabled_count": 1},
                "thresholds": {"min_samples": 25},
                "policy": {"disabled_symbols": ["MSFT"]},
                "symbols": [
                    {
                        "symbol": "MSFT",
                        "effective_mode": "disabled",
                        "quality_score": -50.0,
                        "sample_count": 30,
                    },
                    {
                        "symbol": "AAPL",
                        "effective_mode": "allow",
                        "quality_score": 10.0,
                        "sample_count": 30,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_SYMBOL_UNIVERSE_SCORECARD_PATH", str(artifact))

    snapshot = health_payload._symbol_universe_scorecard_snapshot()

    assert snapshot["available"] is True
    assert snapshot["path"] == str(artifact)
    assert snapshot["summary"]["disabled_count"] == 1
    assert snapshot["policy"]["disabled_symbols"] == ["MSFT"]
    assert snapshot["top_symbols"][0]["symbol"] == "AAPL"
    assert snapshot["bottom_symbols"][0]["symbol"] == "MSFT"


def test_runtime_decay_controls_snapshot_reads_actions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "runtime_decay_controls_latest.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifact_type": "runtime_decay_controls",
                "generated_at": "2026-05-01T15:00:00Z",
                "status": {"available": True, "status": "ready", "mode": "enforce"},
                "observed": {"live_cost_breach_count": 1},
                "actions": {
                    "max_action": "disable_new_entries",
                    "entries_allowed": False,
                    "size_scale": 0.0,
                    "reasons": ["live_cost_breach"],
                },
                "recovery": {"reversible": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_RUNTIME_DECAY_CONTROLS_PATH", str(artifact))

    snapshot = health_payload._runtime_decay_controls_snapshot()

    assert snapshot["available"] is True
    assert snapshot["path"] == str(artifact)
    assert snapshot["actions"]["entries_allowed"] is False
    assert snapshot["actions"]["max_action"] == "disable_new_entries"
    assert snapshot["observed"]["live_cost_breach_count"] == 1
    assert snapshot["recovery"]["reversible"] is True
    assert snapshot["age_seconds"] is not None


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
