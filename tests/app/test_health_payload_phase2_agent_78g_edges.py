from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ai_trading.health_payload as health_payload


def _stub_runtime_components(monkeypatch, *, provider: dict[str, Any], broker: dict[str, Any], service: dict[str, Any]) -> None:
    monkeypatch.setattr(health_payload.runtime_state, "observe_data_provider_state", lambda: provider)
    monkeypatch.setattr(health_payload.runtime_state, "observe_broker_status", lambda: broker)
    monkeypatch.setattr(health_payload.runtime_state, "observe_service_status", lambda: service)
    monkeypatch.setattr(health_payload.runtime_state, "observe_quote_status", lambda: {"status": "stale"})
    monkeypatch.setattr(health_payload, "get_backup_data_provider", lambda: "backup-feed")
    monkeypatch.setattr(health_payload, "_model_liveness_snapshot", lambda: {"ok": False, "reason": "stale_model"})
    monkeypatch.setattr(
        health_payload,
        "_database_readiness_snapshot_cached",
        lambda: {"enabled": True, "configured": False, "ok": True},
    )
    monkeypatch.setattr(health_payload, "_oms_invariants_snapshot_cached", lambda: {"enabled": False})
    monkeypatch.setattr(health_payload, "_oms_lifecycle_parity_snapshot_cached", lambda: {"enabled": False})
    monkeypatch.setattr(
        health_payload,
        "_replay_live_parity_gate_snapshot_cached",
        lambda *, oms_lifecycle_parity=None: {"enabled": False, "ok": True},
    )
    monkeypatch.setattr(health_payload, "_env_bool", lambda _name, default: bool(default))
    monkeypatch.setattr(health_payload, "_env_float", lambda _name, default: float(default))


def test_runtime_payload_normalizes_degraded_provider_broker_and_service(monkeypatch) -> None:
    _stub_runtime_components(
        monkeypatch,
        provider={
            "using_backup": True,
            "safe_mode": True,
            "reason": "provider_timeout",
            "reason_code": "timeout",
            "reason_detail": "primary data feed timed out",
            "http_code": 503,
            "data_status": "empty",
            "gap_ratio_recent": "not-a-ratio",
            "quote_fresh_ms": 90000,
            "cooldown_sec": 12,
        },
        broker={
            "connected": False,
            "last_error": "broker tcp reset",
            "open_orders_count": "4",
            "positions_count": "2",
        },
        service={
            "status": "stopped",
            "reason": "trade_updates_stream_exited",
            "phase": "run",
            "cycle_index": 9,
        },
    )

    payload = health_payload.build_runtime_health_payload(
        service_name="edge-svc",
        healthy_status_mode="healthy",
        ok_mode="connectivity",
    )

    assert payload["service"] == "edge-svc"
    assert payload["ok"] is False
    assert payload["status"] == "degraded"
    assert payload["reason"] == "trade_updates_stream_exited"
    assert payload["http_code"] == 503
    assert payload["fallback_active"] is True
    assert payload["data_provider"]["status"] == "degraded"
    assert payload["data_provider"]["backup"] == "backup-feed"
    assert payload["data_provider"]["gap_ratio_pct"] is None
    assert payload["broker"]["status"] == "unreachable"
    assert payload["broker"]["connected"] is False
    assert payload["service_state"]["phase"] == "run"
    assert payload["quotes_status"] == {"status": "stale"}
    assert payload["model_liveness"] == {"ok": False, "reason": "stale_model"}
    assert payload["readiness_failures"] == []
    assert payload["attention_flags"] == [
        "provider_backup_active",
        "provider_safe_mode",
        "service_degraded",
        "service_halt_active",
        "trade_updates_stream_degraded",
    ]


def test_control_plane_snapshot_preserves_optional_gate_failures_and_normalizes_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = {"status": "ready", "reason": "market_closed", "using_backup": False}
    broker = {"status": "connected", "connected": True, "open_orders_count": "3", "positions_count": "2"}
    service = {"status": "halted", "reason": "hard_stop", "phase": "guarded", "cycle_index": 4}
    _stub_runtime_components(monkeypatch, provider=provider, broker=broker, service=service)
    monkeypatch.setattr(
        health_payload,
        "_database_readiness_snapshot",
        lambda: {"enabled": True, "configured": True, "ok": False},
    )
    monkeypatch.setattr(
        health_payload,
        "_oms_invariants_snapshot",
        lambda: {"enabled": True, "available": True, "ok": False, "reason": "intent_drift", "total_violations": 5},
    )
    monkeypatch.setattr(
        health_payload,
        "_oms_lifecycle_parity_snapshot",
        lambda: {"enabled": True, "available": True, "ok": False, "reason": "missing_terminal", "total_violations": 6},
    )
    monkeypatch.setattr(
        health_payload,
        "_replay_live_parity_gate_snapshot",
        lambda *, oms_lifecycle_parity=None: {"enabled": True, "available": True, "ok": False, "reason": "replay_gap"},
    )
    monkeypatch.setattr(
        health_payload,
        "_runtime_performance_snapshot",
        lambda: {
            "available": True,
            "source": "fixture",
            "broker_open_position_snapshots": {"AAPL": {"qty": 1}},
            "go_no_go": {
                "gate_passed": False,
                "failed_checks": ["positions"],
                "observed": {
                    "open_position_reconciliation_available": True,
                    "open_position_reconciliation_consistent": False,
                    "open_position_reconciliation_ratio": 0.5,
                    "open_position_reconciliation_mismatch_count": 2,
                    "open_position_reconciliation_max_abs_delta_qty": 3,
                    "event_tca_parent_retry_per_order": 1.25,
                    "event_tca_parent_failed_slices_per_order": 0.25,
                    "event_tca_parent_avg_success_ratio": 0.8,
                    "event_tca_parent_avg_arrival_slippage_bps": 4.2,
                    "event_tca_parent_execution_consistent": False,
                    "event_tca_parent_scope_threshold_breach_count": 2,
                },
            },
            "oms_event_tca": {
                "available": True,
                "submit_reject_rate_pct": 7.5,
                "cancel_to_submit_ack_rate_pct": 2.5,
                "reject_cancel_rate_pct": 1.5,
                "p90_slippage_bps": 9.0,
                "parent_execution_summary_events": 12,
                "parent_execution_kpis_by_scope": [{"scope": "core"}, "bad", {"scope": "extended"}],
                "event_outcomes_by_scope": [{"outcome": "filled"}, None],
                "submit_reject_reasons_top": [{"reason": "min_notional"}, []],
                "cancel_reasons_top": [{"reason": "timeout"}, "bad"],
                "realized_slippage_decomposition": {"arrival_bps": 4.2},
            },
        },
    )
    monkeypatch.setattr(health_payload, "_manual_override_snapshot", lambda: {"available": True, "state": {"halt": True}})

    governance_base = tmp_path / "governance"
    governance_base.mkdir()
    (governance_base / "promotion_approvals.jsonl").write_text(
        "\n".join(
            [
                "",
                "{bad-json",
                json.dumps({"id": "old"}),
                json.dumps({"id": "latest", "approved": False}),
            ]
        ),
        encoding="utf-8",
    )
    (governance_base / "champion_challenger_scorecards.jsonl").write_text(
        json.dumps({"model": "challenger", "score": 0.42}),
        encoding="utf-8",
    )
    monkeypatch.setattr(health_payload, "_governance_base_path", lambda: governance_base)

    payload = health_payload.build_control_plane_snapshot(service_name="operator")

    assert payload["service"] == "operator"
    assert payload["rollout"] == {
        "phase": "guarded",
        "phase_since": None,
        "cycle_index": 4,
        "status": "halted",
        "reason": "hard_stop",
    }
    assert payload["attention_flags"] == [
        "market_closed_non_flat_positions",
        "market_closed_open_orders",
        "service_degraded",
        "service_halt_active",
        "replay_live_parity_gate_failed",
        "database_unhealthy",
        "oms_invariants_failed",
        "oms_lifecycle_parity_failed",
    ]
    assert payload["positions"]["broker_position_snapshots"] == {"AAPL": {"qty": 1}}
    assert payload["positions"]["mismatch_count"] == 2
    assert payload["execution_quality"]["parent_execution_kpis_by_scope"] == [
        {"scope": "core"},
        {"scope": "extended"},
    ]
    assert payload["execution_quality"]["event_outcomes_by_scope"] == [{"outcome": "filled"}]
    assert payload["execution_quality"]["submit_reject_reasons_top"] == [{"reason": "min_notional"}]
    assert payload["execution_quality"]["cancel_reasons_top"] == [{"reason": "timeout"}]
    assert payload["circuit_breakers"] == {"go_no_go_gate_passed": False, "failed_checks": ["positions"]}
    assert payload["manual_overrides"] == {"available": True, "state": {"halt": True}}
    assert payload["governance"]["latest_promotion_approval"] == {"id": "latest", "approved": False}
    assert payload["governance"]["latest_champion_challenger_scorecard"] == {
        "model": "challenger",
        "score": 0.42,
    }
    assert payload["governance"]["latest_rollback_audit"] is None
