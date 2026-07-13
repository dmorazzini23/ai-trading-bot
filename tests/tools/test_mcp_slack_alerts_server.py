from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tools import mcp_slack_alerts_server as slack_srv


def test_evaluate_incident_triggers_catches_regressions() -> None:
    snapshot = {
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["profit_factor"],
        "execution_capture_ratio": 0.03,
        "slippage_drag_bps": 8.1,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "runtime_gate_failed",
        "provider_status": "degraded",
        "provider_active": "alpaca",
        "provider_reason": "runtime_gate_failed",
        "using_backup": True,
        "broker_status": "disconnected",
    }
    triggers = slack_srv._evaluate_incident_triggers(snapshot, {"min_capture_ratio": 0.08})
    assert "go_no_go_failed" in triggers
    assert "go_no_go_failed_checks" in triggers
    assert "execution_capture_ratio_low" in triggers
    assert "health_degraded" in triggers
    assert "data_provider_backup_active" in triggers
    assert "broker_disconnected" in triggers


def test_runtime_incident_snapshot_exposes_stable_signature(monkeypatch) -> None:
    snapshot = {
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "runtime_gate_failed",
        "provider_status": "degraded",
        "provider_active": "alpaca",
        "provider_reason": "runtime_gate_failed",
        "using_backup": False,
        "broker_status": "connected",
    }

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", lambda _args: snapshot)

    result = slack_srv.tool_runtime_incident_snapshot({})

    assert result["should_alert"] is True
    assert result["triggers"] == ["health_degraded"]
    assert result["incident_signature"] == slack_srv._incident_signature(
        snapshot,
        ["health_degraded"],
    )
    assert result["incident_signature"] != result["fingerprint"]


def test_evaluate_incident_triggers_flags_fill_and_precheck_spikes() -> None:
    snapshot = {
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "execution_fill_ratio": 0.18,
        "execution_fill_ratio_samples": 30,
        "execution_fill_ratio_filled": 5,
        "execution_window_minutes": 30,
        "execution_skipped_count": 20,
        "precheck_failure_count": 15,
        "precheck_failure_ratio": 0.75,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "runtime_health_ok",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "data_available_netting",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "min_fill_ratio": 0.25,
            "min_fill_ratio_samples": 20,
            "precheck_spike_min_count": 10,
            "precheck_spike_min_ratio": 0.6,
            "precheck_spike_min_skipped": 12,
        },
    )
    assert "execution_fill_ratio_low" in triggers
    assert "pre_execution_checks_spike" in triggers


def test_evaluate_incident_triggers_suppresses_gonogo_when_openings_not_blocked() -> None:
    snapshot = {
        "runtime_gonogo_block_openings_enabled": False,
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["profit_factor", "win_rate"],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "runtime_health_ok",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "data_available_netting",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(snapshot, {"min_capture_ratio": 0.08})
    assert "go_no_go_failed" not in triggers
    assert "go_no_go_failed_checks" not in triggers


def test_evaluate_incident_triggers_suppresses_gonogo_when_market_closed() -> None:
    snapshot = {
        "runtime_gonogo_block_openings_enabled": True,
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["win_rate", "acceptance_rate", "live_samples_sufficient"],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "market_closed",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(snapshot, {"min_capture_ratio": 0.08})
    assert "go_no_go_failed" not in triggers
    assert "go_no_go_failed_checks" not in triggers


def test_evaluate_incident_triggers_can_disable_market_closed_gonogo_suppression() -> None:
    snapshot = {
        "runtime_gonogo_block_openings_enabled": True,
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["win_rate", "acceptance_rate", "live_samples_sufficient"],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "market_closed",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "suppress_market_closed_gonogo_alerts": False,
        },
    )
    assert "go_no_go_failed" in triggers
    assert "go_no_go_failed_checks" in triggers


def test_evaluate_incident_triggers_suppresses_gonogo_during_startup_grace() -> None:
    now = datetime.now(UTC)
    snapshot = {
        "runtime_gonogo_block_openings_enabled": True,
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["win_rate", "acceptance_rate"],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "runtime_health_ok",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "data_available_netting",
        "using_backup": False,
        "broker_status": "connected",
        "service_phase_since": (now - timedelta(seconds=60)).isoformat(),
        "timestamp": now.isoformat(),
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {"min_capture_ratio": 0.08, "startup_grace_seconds": 300},
    )
    assert "go_no_go_failed" not in triggers
    assert "go_no_go_failed_checks" not in triggers


def test_evaluate_incident_triggers_suppresses_precheck_spike_when_fill_ratio_healthy() -> None:
    snapshot = {
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "execution_fill_ratio": 0.62,
        "execution_fill_ratio_samples": 40,
        "execution_fill_ratio_filled": 25,
        "execution_window_minutes": 30,
        "execution_skipped_count": 20,
        "precheck_failure_count": 18,
        "precheck_failure_ratio": 0.9,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "runtime_health_ok",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "data_available_netting",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "min_fill_ratio": 0.25,
            "min_fill_ratio_samples": 20,
            "precheck_spike_min_count": 10,
            "precheck_spike_min_ratio": 0.6,
            "precheck_spike_min_skipped": 12,
        },
    )
    assert "execution_fill_ratio_low" not in triggers
    assert "pre_execution_checks_spike" not in triggers


def test_evaluate_incident_triggers_flags_rejection_concentration_and_realism_gap() -> None:
    snapshot = {
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "expected_edge_per_accept_bps": 8.0,
        "realization_gap_bps": -6.0,
        "edge_realism_gap_ratio": 0.25,
        "gate_rejected_records": 40,
        "top_rejection_concentration_gate": "symbol_live_expectancy_gate",
        "top_rejection_concentration_ratio": 0.8,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "runtime_health_ok",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "data_available_netting",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "max_rejection_concentration_ratio": 0.65,
            "min_rejected_records_for_concentration": 20,
            "min_expected_edge_bps_for_realism": 0.5,
            "min_edge_realism_ratio": 0.35,
        },
    )
    assert "rejection_concentration_high" in triggers
    assert "edge_realism_gap_high" in triggers


def test_evaluate_incident_triggers_suppresses_edge_realism_when_market_closed() -> None:
    snapshot = {
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "expected_edge_per_accept_bps": 8.0,
        "realization_gap_bps": -6.0,
        "edge_realism_gap_ratio": 0.2,
        "gate_rejected_records": 0,
        "top_rejection_concentration_ratio": None,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "market_closed",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "min_expected_edge_bps_for_realism": 0.5,
            "min_edge_realism_ratio": 0.35,
        },
    )
    assert "edge_realism_gap_high" not in triggers


def test_evaluate_incident_triggers_suppresses_performance_spikes_when_market_closed() -> None:
    snapshot = {
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.01,
        "slippage_drag_bps": 7.0,
        "execution_fill_ratio": 0.05,
        "execution_fill_ratio_samples": 40,
        "execution_skipped_count": 30,
        "precheck_failure_count": 25,
        "precheck_failure_ratio": 0.83,
        "gate_rejected_records": 60,
        "top_rejection_concentration_ratio": 0.9,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "market_closed",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "min_fill_ratio": 0.25,
            "min_fill_ratio_samples": 20,
            "precheck_spike_min_count": 10,
            "precheck_spike_min_ratio": 0.6,
            "precheck_spike_min_skipped": 12,
            "max_rejection_concentration_ratio": 0.65,
            "min_rejected_records_for_concentration": 20,
        },
    )
    assert "execution_capture_ratio_low" not in triggers
    assert "execution_fill_ratio_low" not in triggers
    assert "pre_execution_checks_spike" not in triggers
    assert "rejection_concentration_high" not in triggers


def test_evaluate_incident_triggers_suppresses_fill_and_precheck_during_startup() -> None:
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "execution_fill_ratio": 0.05,
        "execution_fill_ratio_samples": 50,
        "execution_skipped_count": 30,
        "precheck_failure_count": 25,
        "precheck_failure_ratio": 0.83,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "startup_pending_reconcile_complete",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "execution_quote_ready",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "min_fill_ratio": 0.25,
            "min_fill_ratio_samples": 20,
            "precheck_spike_min_count": 10,
            "precheck_spike_min_ratio": 0.6,
            "precheck_spike_min_skipped": 12,
        },
    )
    assert "execution_fill_ratio_low" not in triggers
    assert "pre_execution_checks_spike" not in triggers


def test_evaluate_incident_triggers_treats_reachable_broker_as_healthy() -> None:
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "runtime_health_ok",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "data_available_netting",
        "using_backup": False,
        "broker_status": "reachable",
    }
    triggers = slack_srv._evaluate_incident_triggers(snapshot, {"min_capture_ratio": 0.08})
    assert "broker_disconnected" not in triggers


def test_evaluate_incident_triggers_suppresses_startup_warmup_health() -> None:
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "warmup_cycle",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(snapshot, {"min_capture_ratio": 0.08})
    assert "health_degraded" not in triggers


def test_evaluate_incident_triggers_can_disable_startup_warmup_suppression() -> None:
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "warmup_cycle",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "suppress_startup_warmup_health_alerts": False,
        },
    )
    assert "health_degraded" in triggers


def test_evaluate_incident_triggers_suppresses_transient_startup_health_with_grace() -> None:
    now = datetime.now(UTC)
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "broker_status_unknown",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "startup_config_resolved",
        "using_backup": False,
        "broker_status": "unknown",
        "service_phase_since": (now - timedelta(seconds=60)).isoformat(),
        "timestamp": now.isoformat(),
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {"min_capture_ratio": 0.08, "startup_grace_seconds": 300},
    )
    assert "health_degraded" not in triggers


def test_evaluate_incident_triggers_suppresses_restart_bootstrap_provider_unknown() -> None:
    now = datetime.now(UTC)
    snapshot = {
        "runtime_gonogo_block_openings_enabled": True,
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["win_rate", "live_samples_sufficient"],
        "expected_edge_per_accept_bps": 30.55,
        "edge_realism_gap_ratio": 0.25,
        "execution_capture_ratio": 0.49,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "provider_status_unknown",
        "provider_status": "unknown",
        "provider_active": "unknown",
        "provider_reason": "unknown",
        "using_backup": False,
        "broker_status": "unknown",
        "service_status": "warming_up",
        "service_phase": "bootstrap",
        "service_reason": "startup",
        "service_phase_since": (now - timedelta(seconds=13)).isoformat(),
        "timestamp": now.isoformat(),
    }

    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "startup_grace_seconds": 300,
            "min_capture_ratio": 0.08,
            "min_edge_realism_ratio": 0.35,
            "min_expected_edge_bps_for_realism": 0.5,
        },
    )

    assert triggers == []


def test_evaluate_incident_triggers_allows_degraded_health_after_startup_grace() -> None:
    now = datetime.now(UTC)
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "broker_status_unknown",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "startup_config_resolved",
        "using_backup": False,
        "broker_status": "unknown",
        "service_phase_since": (now - timedelta(seconds=600)).isoformat(),
        "timestamp": now.isoformat(),
    }
    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {"min_capture_ratio": 0.08, "startup_grace_seconds": 300},
    )
    assert "health_degraded" in triggers


def test_evaluate_incident_triggers_suppresses_market_closed_readiness_only_health() -> None:
    snapshot = {
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["replay_live_parity_gate_failed"],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": False,
        "health_status": "healthy",
        "health_reason": "market_closed",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "connected",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {
            "min_capture_ratio": 0.08,
            "suppress_market_closed_gonogo_alerts": True,
            "suppress_market_closed_health_alerts": True,
        },
    )

    assert "health_degraded" not in triggers
    assert "go_no_go_failed" not in triggers


def test_evaluate_incident_triggers_allows_market_closed_broker_health_incident() -> None:
    snapshot = {
        "go_no_go_gate_passed": None,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": None,
        "slippage_drag_bps": None,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "market_closed",
        "provider_status": "warming_up",
        "provider_active": "alpaca",
        "provider_reason": "market_closed",
        "using_backup": False,
        "broker_status": "disconnected",
        "timestamp": datetime.now(UTC).isoformat(),
    }

    triggers = slack_srv._evaluate_incident_triggers(
        snapshot,
        {"min_capture_ratio": 0.08, "suppress_market_closed_health_alerts": True},
    )

    assert "health_degraded" in triggers
    assert "broker_disconnected" in triggers


def test_collect_execution_window_snapshot_tracks_precheck_detail_breakdown(
    monkeypatch, tmp_path: Path
) -> None:
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    quality_path = tmp_path / "execution_quality_events.jsonl"
    quality_path.write_text(
        "\n".join(
            [
                json_line
                for json_line in [
                    (
                        '{"ts":"%s","status":"skipped","reason":"pre_execution_order_checks_failed",'
                        '"detail":"symbol_reentry_cooldown"}'
                    )
                    % now_iso,
                    (
                        '{"ts":"%s","status":"skipped","reason":"pre_execution_order_checks_failed",'
                        '"detail":"opening_min_notional"}'
                    )
                    % now_iso,
                    (
                        '{"ts":"%s","status":"skipped","reason":"pre_execution_order_checks_failed",'
                        '"detail":"symbol_reentry_cooldown"}'
                    )
                    % now_iso,
                    (
                        '{"ts":"%s","status":"skipped","reason":"execution_phase_gate"}'
                    )
                    % now_iso,
                ]
            ]
        )
        + "\n"
    )
    order_path = tmp_path / "order_events.jsonl"
    order_path.write_text("")
    monkeypatch.setenv("AI_TRADING_EXEC_QUALITY_EVENTS_PATH", str(quality_path))
    monkeypatch.setenv("AI_TRADING_ORDER_EVENTS_PATH", str(order_path))

    snapshot = slack_srv._collect_execution_window_snapshot({})

    assert snapshot["execution_skipped_count"] == 4
    assert snapshot["precheck_failure_count"] == 3
    assert snapshot["precheck_failure_top_details"] == [
        {"detail": "symbol_reentry_cooldown", "count": 2},
        {"detail": "opening_min_notional", "count": 1},
    ]
    assert snapshot["precheck_failure_top_actionable_details"] == [
        {"detail": "symbol_reentry_cooldown", "count": 2},
        {"detail": "opening_min_notional", "count": 1},
    ]


def test_collect_gate_window_snapshot_uses_blocking_only_concentration_gate(
    monkeypatch, tmp_path: Path
) -> None:
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    gate_path = tmp_path / "gate_effectiveness.jsonl"
    gate_path.write_text(
        json.dumps(
            {
                "ts": now_iso,
                "rejected_records": 10,
                "gate_attribution": {
                    "EXPECTED_CAPTURE_MODEL_LEARNED": {
                        "blocked_records": 9,
                        "accepted_records": 5,
                    },
                    "PASSIVE_FILL_PROBABILITY_LOW": {
                        "blocked_records": 3,
                        "accepted_records": 0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_LOG_PATH", str(gate_path))

    snapshot = slack_srv._collect_gate_window_snapshot(
        {"gate_window_minutes": 60, "gate_window_max_rows": 200}
    )

    assert snapshot["top_rejection_concentration_gate"] == "PASSIVE_FILL_PROBABILITY_LOW"
    assert snapshot["top_rejection_concentration_blocking_gate_found"] is True
    assert abs(float(snapshot["top_rejection_concentration_ratio"] or 0.0) - 0.3) < 1e-9


def test_collect_gate_window_snapshot_excludes_non_blocking_model_tags(
    monkeypatch, tmp_path: Path
) -> None:
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    gate_path = tmp_path / "gate_effectiveness.jsonl"
    gate_path.write_text(
        json.dumps(
            {
                "ts": now_iso,
                "rejected_records": 10,
                "gate_attribution": {
                    "EXPECTED_CAPTURE_MODEL_LEARNED": {
                        "blocked_records": 9,
                        "accepted_records": 0,
                    },
                    "PRE_EXECUTION_ORDER_CHECKS_FAILED": {
                        "blocked_records": 3,
                        "accepted_records": 0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_LOG_PATH", str(gate_path))

    snapshot = slack_srv._collect_gate_window_snapshot(
        {"gate_window_minutes": 60, "gate_window_max_rows": 200}
    )

    assert snapshot["top_rejection_concentration_gate"] == "PRE_EXECUTION_ORDER_CHECKS_FAILED"
    assert snapshot["top_rejection_concentration_blocking_gate_found"] is True
    assert abs(float(snapshot["top_rejection_concentration_ratio"] or 0.0) - 0.3) < 1e-9


def test_collect_runtime_snapshot_does_not_fallback_to_report_concentration_for_annotation_only_window(
    monkeypatch, tmp_path: Path
) -> None:
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    gate_path = tmp_path / "gate_effectiveness.jsonl"
    gate_path.write_text(
        json.dumps(
            {
                "ts": now_iso,
                "rejected_records": 12,
                "gate_attribution": {
                    "EXPECTED_CAPTURE_MODEL_LEARNED": {
                        "blocked_records": 11,
                        "accepted_records": 4,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_TRADING_GATE_EFFECTIVENESS_LOG_PATH", str(gate_path))
    monkeypatch.setattr(
        slack_srv,
        "_runtime_report_payload",
        lambda: {
            "go_no_go": {"gate_passed": True, "failed_checks": []},
            "execution_vs_alpha": {},
            "gate_effectiveness": {
                "rejected_records": 999,
                "top_rejection_concentration_gate": "EXPECTED_CAPTURE_MODEL_LEARNED",
                "top_rejection_concentration_ratio": 0.99,
            },
        },
    )
    monkeypatch.setattr(
        slack_srv,
        "_health_payload",
        lambda **_: {
            "status": "healthy",
            "reason": "runtime_health_ok",
            "timestamp": now_iso,
            "data_provider": {
                "status": "healthy",
                "active": "alpaca",
                "using_backup": False,
                "reason": "data_available_netting",
            },
            "broker": {"status": "connected"},
        },
    )
    monkeypatch.setattr(
        slack_srv,
        "_collect_execution_window_snapshot",
        lambda _args: {
            "execution_fill_ratio": None,
            "execution_fill_ratio_samples": 0,
            "execution_fill_ratio_filled": 0,
            "execution_window_minutes": 30,
            "execution_skipped_count": 0,
            "precheck_failure_count": 0,
            "precheck_failure_ratio": None,
            "precheck_failure_top_details": [],
            "precheck_failure_top_actionable_details": [],
            "order_events_path": str(tmp_path / "order_events.jsonl"),
            "exec_quality_events_path": str(tmp_path / "execution_quality_events.jsonl"),
        },
    )

    snapshot = slack_srv._collect_runtime_snapshot({})

    assert snapshot["gate_rejected_records"] == 12
    assert snapshot["top_rejection_concentration_gate"] == ""
    assert snapshot["top_rejection_concentration_ratio"] is None


def test_collect_runtime_snapshot_ignores_ok_trade_report_concentration(
    monkeypatch, tmp_path: Path
) -> None:
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    monkeypatch.setattr(
        slack_srv,
        "_runtime_report_payload",
        lambda: {
            "go_no_go": {"gate_passed": True, "failed_checks": []},
            "execution_vs_alpha": {},
            "gate_effectiveness": {
                "rejected_records": 20,
                "top_rejection_concentration_gate": "OK_TRADE",
                "top_rejection_concentration_ratio": 0.9,
            },
        },
    )
    monkeypatch.setattr(
        slack_srv,
        "_health_payload",
        lambda **_: {
            "status": "healthy",
            "reason": "runtime_health_ok",
            "timestamp": now_iso,
            "data_provider": {
                "status": "healthy",
                "active": "alpaca",
                "using_backup": False,
                "reason": "data_available_netting",
            },
            "broker": {"status": "connected"},
        },
    )
    monkeypatch.setattr(
        slack_srv,
        "_collect_execution_window_snapshot",
        lambda _args: {
            "execution_fill_ratio": None,
            "execution_fill_ratio_samples": 0,
            "execution_fill_ratio_filled": 0,
            "execution_window_minutes": 30,
            "execution_skipped_count": 0,
            "precheck_failure_count": 0,
            "precheck_failure_ratio": None,
            "precheck_failure_top_details": [],
            "precheck_failure_top_actionable_details": [],
            "order_events_path": str(tmp_path / "order_events.jsonl"),
            "exec_quality_events_path": str(tmp_path / "execution_quality_events.jsonl"),
        },
    )
    monkeypatch.setattr(
        slack_srv,
        "_collect_gate_window_snapshot",
        lambda _args: {
            "gate_rejected_records": 0,
            "gate_window_rows": 0,
            "gate_window_minutes": 60,
            "gate_window_events_path": str(tmp_path / "gate_effectiveness.jsonl"),
            "top_rejection_concentration_blocking_gate_found": False,
        },
    )

    snapshot = slack_srv._collect_runtime_snapshot({})

    assert snapshot["gate_rejected_records"] == 20
    assert snapshot["top_rejection_concentration_gate"] == ""
    assert snapshot["top_rejection_concentration_ratio"] is None


def test_collect_runtime_snapshot_degrades_when_health_payload_fails(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        slack_srv,
        "_runtime_report_payload",
        lambda: {
            "go_no_go": {"gate_passed": True, "failed_checks": []},
            "execution_vs_alpha": {"execution_capture_ratio": 0.25},
            "gate_effectiveness": {},
        },
    )
    monkeypatch.setattr(
        slack_srv,
        "_health_payload",
        lambda **_: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )
    monkeypatch.setattr(
        slack_srv,
        "_collect_execution_window_snapshot",
        lambda _args: {
            "execution_fill_ratio": None,
            "execution_fill_ratio_samples": 0,
            "execution_fill_ratio_filled": 0,
            "execution_window_minutes": 30,
            "execution_skipped_count": 0,
            "precheck_failure_count": 0,
            "precheck_failure_ratio": None,
            "precheck_failure_top_details": [],
            "precheck_failure_top_actionable_details": [],
            "order_events_path": str(tmp_path / "order_events.jsonl"),
            "exec_quality_events_path": str(tmp_path / "execution_quality_events.jsonl"),
        },
    )
    monkeypatch.setattr(
        slack_srv,
        "_collect_gate_window_snapshot",
        lambda _args: {
            "gate_rejected_records": 0,
            "gate_window_rows": 0,
            "gate_window_minutes": 60,
            "gate_window_events_path": str(tmp_path / "gate_effectiveness.jsonl"),
            "top_rejection_concentration_blocking_gate_found": False,
        },
    )

    snapshot = slack_srv._collect_runtime_snapshot({})

    assert snapshot["health_ok"] is False
    assert snapshot["health_status"] == "degraded"
    assert snapshot["health_reason"] == "health_payload_unavailable"
    assert snapshot["provider_status"] == "unknown"
    assert snapshot["broker_status"] == "unknown"


def test_incident_message_includes_top_precheck_blockers() -> None:
    snapshot = {
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "execution_fill_ratio": 0.6,
        "execution_fill_ratio_samples": 50,
        "execution_fill_ratio_filled": 30,
        "execution_window_minutes": 30,
        "execution_skipped_count": 20,
        "precheck_failure_count": 15,
        "precheck_failure_ratio": 0.75,
        "precheck_failure_top_details": [
            {"detail": "symbol_reentry_cooldown", "count": 9},
            {"detail": "opening_min_notional", "count": 6},
        ],
        "health_ok": True,
        "health_status": "healthy",
        "health_reason": "runtime_health_ok",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "provider_reason": "data_available_netting",
        "using_backup": False,
        "broker_status": "connected",
        "timestamp": "2026-04-02T19:00:00Z",
    }

    text = slack_srv._incident_message_text(snapshot, ["pre_execution_checks_spike"])

    assert "Top pre-check blockers (30m): symbol reentry cooldown=9, opening min notional=6" in text


def test_notify_incident_channel_dedupes(monkeypatch, tmp_path: Path) -> None:
    snapshot = {
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["win_rate"],
        "execution_capture_ratio": 0.05,
        "slippage_drag_bps": 7.0,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "runtime_gate_failed",
        "provider_status": "degraded",
        "provider_active": "alpaca",
        "provider_reason": "runtime_gate_failed",
        "using_backup": False,
        "broker_status": "connected",
        "timestamp": "2026-03-28T20:00:00Z",
    }

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    state_path = tmp_path / "slack_state.json"
    args = {
        "state_path": str(state_path),
        "webhook_url": "https://hooks.slack.test/example",
    }
    first = slack_srv.tool_notify_incident_channel(args)
    second = slack_srv.tool_notify_incident_channel(args)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "repeat_cooldown_active"
    assert len(posts) == 1
    payload = posts[0]["payload"]
    assert isinstance(payload, dict)
    assert "ai-trading incident update" in str(payload.get("text"))
    blocks = payload.get("blocks")
    assert isinstance(blocks, list)
    assert blocks[0]["type"] == "header"
    assert any("Triggered by" in str(block) for block in blocks)


def test_notify_incident_channel_confirms_health_unavailable_before_alert(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = {
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["win_rate"],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "health_payload_unavailable",
        "provider_status": "unknown",
        "provider_active": "unknown",
        "provider_reason": "health_payload_unavailable",
        "using_backup": False,
        "broker_status": "unknown",
        "timestamp": "2026-03-28T20:00:00Z",
    }

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    args = {
        "state_path": str(tmp_path / "slack_state.json"),
        "webhook_url": "https://hooks.slack.test/example",
    }
    first = slack_srv.tool_notify_incident_channel(args)
    second = slack_srv.tool_notify_incident_channel(args)

    assert first["sent"] is False
    assert first["reason"] == "health_unavailable_confirmation_pending"
    assert second["sent"] is True
    assert len(posts) == 1


def test_notify_incident_channel_can_disable_health_unavailable_confirmation(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = {
        "go_no_go_gate_passed": False,
        "go_no_go_failed_checks": ["win_rate"],
        "execution_capture_ratio": 0.2,
        "slippage_drag_bps": 7.0,
        "health_ok": False,
        "health_status": "degraded",
        "health_reason": "health_payload_unavailable",
        "provider_status": "unknown",
        "provider_active": "unknown",
        "provider_reason": "health_payload_unavailable",
        "using_backup": False,
        "broker_status": "unknown",
        "timestamp": "2026-03-28T20:00:00Z",
    }
    posts: list[dict[str, object]] = []

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", lambda _args: snapshot)
    monkeypatch.setattr(
        slack_srv,
        "_post_slack_message",
        lambda webhook_url, payload, timeout_s=5.0: posts.append(
            {"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s}
        )
        or 200,
    )

    result = slack_srv.tool_notify_incident_channel(
        {
            "state_path": str(tmp_path / "slack_state.json"),
            "webhook_url": "https://hooks.slack.test/example",
            "confirm_health_unavailable_before_alert": False,
        }
    )

    assert result["sent"] is True
    assert len(posts) == 1


def test_notify_incident_channel_dedupes_on_signature_drift(monkeypatch, tmp_path: Path) -> None:
    snapshots = [
        {
            "go_no_go_gate_passed": True,
            "go_no_go_failed_checks": [],
            "execution_capture_ratio": 0.12,
            "slippage_drag_bps": 7.0,
            "execution_fill_ratio": 0.22,
            "execution_fill_ratio_samples": 30,
            "execution_fill_ratio_filled": 7,
            "execution_window_minutes": 30,
            "execution_skipped_count": 15,
            "precheck_failure_count": 14,
            "precheck_failure_ratio": 0.8,
            "health_ok": True,
            "health_status": "healthy",
            "health_reason": "runtime_health_ok",
            "provider_status": "healthy",
            "provider_active": "alpaca",
            "provider_reason": "data_available_netting",
            "using_backup": False,
            "broker_status": "connected",
            "timestamp": "2026-03-28T20:00:00Z",
        },
        {
            "go_no_go_gate_passed": True,
            "go_no_go_failed_checks": [],
            "execution_capture_ratio": 0.121,
            "slippage_drag_bps": 7.01,
            "execution_fill_ratio": 0.219,
            "execution_fill_ratio_samples": 31,
            "execution_fill_ratio_filled": 7,
            "execution_window_minutes": 30,
            "execution_skipped_count": 16,
            "precheck_failure_count": 15,
            "precheck_failure_ratio": 0.82,
            "health_ok": True,
            "health_status": "healthy",
            "health_reason": "runtime_health_ok",
            "provider_status": "healthy",
            "provider_active": "alpaca",
            "provider_reason": "data_available_netting",
            "using_backup": False,
            "broker_status": "connected",
            "timestamp": "2026-03-28T20:05:00Z",
        },
    ]

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        idx = min(len(posts), len(snapshots) - 1)
        return snapshots[idx]

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    state_path = tmp_path / "slack_state.json"
    args = {
        "state_path": str(state_path),
        "webhook_url": "https://hooks.slack.test/example",
        "repeat_cooldown_minutes": 120,
    }
    first = slack_srv.tool_notify_incident_channel(args)
    second = slack_srv.tool_notify_incident_channel(args)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "repeat_cooldown_active"
    assert len(posts) == 1


def test_notify_incident_channel_dedupes_provider_reason_churn(
    monkeypatch, tmp_path: Path
) -> None:
    snapshots = [
        {
            "go_no_go_gate_passed": True,
            "go_no_go_failed_checks": [],
            "execution_capture_ratio": 0.12,
            "slippage_drag_bps": 7.0,
            "execution_fill_ratio": 0.22,
            "execution_fill_ratio_samples": 30,
            "execution_fill_ratio_filled": 7,
            "execution_window_minutes": 30,
            "execution_skipped_count": 15,
            "precheck_failure_count": 14,
            "precheck_failure_ratio": 0.8,
            "health_ok": True,
            "health_status": "healthy",
            "health_reason": "runtime_health_ok",
            "provider_status": "healthy",
            "provider_active": "alpaca",
            "provider_reason": "execution_quote_ready",
            "using_backup": False,
            "broker_status": "connected",
            "timestamp": "2026-03-28T20:00:00Z",
        },
        {
            "go_no_go_gate_passed": True,
            "go_no_go_failed_checks": [],
            "execution_capture_ratio": 0.12,
            "slippage_drag_bps": 7.0,
            "execution_fill_ratio": 0.22,
            "execution_fill_ratio_samples": 30,
            "execution_fill_ratio_filled": 7,
            "execution_window_minutes": 30,
            "execution_skipped_count": 15,
            "precheck_failure_count": 14,
            "precheck_failure_ratio": 0.8,
            "health_ok": True,
            "health_status": "healthy",
            "health_reason": "runtime_health_ok",
            "provider_status": "healthy",
            "provider_active": "alpaca",
            "provider_reason": "data_available_netting",
            "using_backup": False,
            "broker_status": "reachable",
            "timestamp": "2026-03-28T20:05:00Z",
        },
    ]

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        idx = min(len(posts), len(snapshots) - 1)
        return snapshots[idx]

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    state_path = tmp_path / "slack_state.json"
    args = {
        "state_path": str(state_path),
        "webhook_url": "https://hooks.slack.test/example",
        "repeat_cooldown_minutes": 120,
    }
    first = slack_srv.tool_notify_incident_channel(args)
    second = slack_srv.tool_notify_incident_channel(args)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "repeat_cooldown_active"
    assert len(posts) == 1


def test_notify_incident_channel_dedupes_health_status_churn_with_default_cooldown(
    monkeypatch, tmp_path: Path
) -> None:
    snapshots = [
        {
            "go_no_go_gate_passed": True,
            "go_no_go_failed_checks": [],
            "execution_capture_ratio": 0.2,
            "slippage_drag_bps": 7.0,
            "health_ok": False,
            "health_status": "degraded",
            "health_reason": "runtime_gate_failed",
            "provider_status": "degraded",
            "provider_active": "alpaca",
            "provider_reason": "runtime_gate_failed",
            "using_backup": False,
            "broker_status": "connected",
            "timestamp": "2026-05-06T15:00:00Z",
        },
        {
            "go_no_go_gate_passed": True,
            "go_no_go_failed_checks": [],
            "execution_capture_ratio": 0.2,
            "slippage_drag_bps": 7.0,
            "health_ok": False,
            "health_status": "unhealthy",
            "health_reason": "runtime_gate_failed",
            "provider_status": "unavailable",
            "provider_active": "alpaca",
            "provider_reason": "runtime_gate_failed",
            "using_backup": False,
            "broker_status": "reachable",
            "timestamp": "2026-05-06T15:01:00Z",
        },
    ]
    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        idx = min(len(posts), len(snapshots) - 1)
        return snapshots[idx]

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    args = {
        "state_path": str(tmp_path / "slack_state.json"),
        "webhook_url": "https://hooks.slack.test/example",
    }
    first = slack_srv.tool_notify_incident_channel(args)
    second = slack_srv.tool_notify_incident_channel(args)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "repeat_cooldown_active"
    assert len(posts) == 1


def test_notify_incident_channel_dedupes_precheck_spike_under_same_gonogo(
    monkeypatch, tmp_path: Path
) -> None:
    snapshots = [
        {
            "runtime_gonogo_block_openings_enabled": True,
            "go_no_go_gate_passed": False,
            "go_no_go_failed_checks": ["win_rate"],
            "execution_capture_ratio": 0.54,
            "slippage_drag_bps": 8.4,
            "execution_fill_ratio": None,
            "execution_fill_ratio_samples": 0,
            "execution_fill_ratio_filled": 0,
            "execution_window_minutes": 30,
            "execution_skipped_count": 29,
            "precheck_failure_count": 29,
            "precheck_failure_ratio": 1.0,
            "health_ok": True,
            "health_status": "ready",
            "health_reason": "runtime_health_ok",
            "provider_status": "healthy",
            "provider_active": "alpaca-iex",
            "provider_reason": "data_available_netting",
            "using_backup": False,
            "broker_status": "connected",
            "timestamp": "2026-05-05T15:30:14Z",
        },
        {
            "runtime_gonogo_block_openings_enabled": True,
            "go_no_go_gate_passed": False,
            "go_no_go_failed_checks": ["win_rate"],
            "execution_capture_ratio": 0.54,
            "slippage_drag_bps": 8.4,
            "execution_fill_ratio": None,
            "execution_fill_ratio_samples": 0,
            "execution_fill_ratio_filled": 0,
            "execution_window_minutes": 30,
            "execution_skipped_count": 31,
            "precheck_failure_count": 31,
            "precheck_failure_ratio": 1.0,
            "health_ok": True,
            "health_status": "ready",
            "health_reason": "runtime_health_ok",
            "provider_status": "healthy",
            "provider_active": "alpaca-iex",
            "provider_reason": "data_available_netting",
            "using_backup": False,
            "broker_status": "connected",
            "timestamp": "2026-05-05T15:31:17Z",
        },
    ]

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        idx = min(len(posts), len(snapshots) - 1)
        return snapshots[idx]

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    args = {
        "state_path": str(tmp_path / "slack_state.json"),
        "webhook_url": "https://hooks.slack.test/example",
        "repeat_cooldown_minutes": 120,
        "precheck_spike_min_count": 30,
        "precheck_spike_min_ratio": 0.75,
        "precheck_spike_min_skipped": 30,
    }
    first = slack_srv.tool_notify_incident_channel(args)
    second = slack_srv.tool_notify_incident_channel(args)

    assert first["sent"] is True
    assert first["triggers"] == ["go_no_go_failed", "go_no_go_failed_checks"]
    assert second["sent"] is False
    assert second["reason"] == "repeat_cooldown_active"
    assert "pre_execution_checks_spike" in second["triggers"]
    assert len(posts) == 1


def test_notify_incident_channel_honors_min_interval(
    monkeypatch, tmp_path: Path
) -> None:
    snapshots = [
        {
            "go_no_go_gate_passed": False,
            "go_no_go_failed_checks": ["win_rate"],
            "execution_capture_ratio": 0.07,
            "slippage_drag_bps": 7.0,
            "health_ok": False,
            "health_status": "degraded",
            "health_reason": "runtime_gate_failed",
            "provider_status": "degraded",
            "provider_active": "alpaca",
            "provider_reason": "runtime_gate_failed",
            "using_backup": False,
            "broker_status": "connected",
            "timestamp": "2026-03-28T20:00:00Z",
        },
        {
            "go_no_go_gate_passed": False,
            "go_no_go_failed_checks": ["win_rate", "slippage_drag_bps"],
            "execution_capture_ratio": 0.06,
            "slippage_drag_bps": 8.0,
            "health_ok": False,
            "health_status": "degraded",
            "health_reason": "runtime_gate_failed",
            "provider_status": "degraded",
            "provider_active": "alpaca",
            "provider_reason": "runtime_gate_failed",
            "using_backup": False,
            "broker_status": "connected",
            "timestamp": "2026-03-28T20:05:00Z",
        },
    ]

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        idx = min(len(posts), len(snapshots) - 1)
        return snapshots[idx]

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_runtime_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    state_path = tmp_path / "slack_state.json"
    args = {
        "state_path": str(state_path),
        "webhook_url": "https://hooks.slack.test/example",
        "repeat_cooldown_minutes": 0,
        "min_interval_minutes": 60,
    }
    first = slack_srv.tool_notify_incident_channel(args)
    second = slack_srv.tool_notify_incident_channel(args)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "min_interval_active"
    assert len(posts) == 1


def test_notify_eod_summary_sends_once_per_report_date(monkeypatch, tmp_path: Path) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "go_no_go_gate_passed": True,
        "go_no_go_failed_checks": [],
        "net_pnl": 123.45,
        "profit_factor": 1.2,
        "win_rate": 0.57,
        "closed_trades": 42,
        "execution_capture_ratio": 0.18,
        "slippage_drag_bps": 7.5,
        "order_reject_rate_pct": 0.0,
        "open_position_reconciliation_mismatch_count": 0,
        "open_position_reconciliation_max_abs_delta_qty": 0.0,
        "health_status": "healthy",
        "health_reason": "market_closed",
        "provider_status": "healthy",
        "provider_active": "alpaca",
        "using_backup": False,
        "broker_status": "connected",
        "top_loss_symbols": [],
        "timestamp": "2026-03-29T01:00:00Z",
        "learning": {
            "after_hours": {"model_name": "xgb", "model_id": "m123"},
            "execution_learning": {"samples": 20},
            "execution_autotune": {"enabled": True},
            "model_liveness": {},
        },
    }

    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    state_path = tmp_path / "slack_eod_state.json"
    args = {
        "state_path": str(state_path),
        "webhook_url": "https://hooks.slack.test/example",
        "require_after_hours_training": False,
    }
    first = slack_srv.tool_notify_eod_summary(args)
    second = slack_srv.tool_notify_eod_summary(args)

    assert first["sent"] is True
    assert second["sent"] is False
    assert second["reason"] == "already_sent_for_report_date"
    assert len(posts) == 1
    payload = posts[0]["payload"]
    assert isinstance(payload, dict)
    text = payload.get("text")
    assert isinstance(text, str)
    assert "ai-trading EOD summary" in text
    blocks = payload.get("blocks")
    assert isinstance(blocks, list)
    assert blocks[0]["type"] == "header"
    assert any("Day performance" in str(block) for block in blocks)


def test_notify_eod_summary_respects_market_closed_gate(monkeypatch) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "health_reason": "runtime_health_ok",
        "learning": {},
    }

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    result = slack_srv.tool_notify_eod_summary(
        {
            "webhook_url": "https://hooks.slack.test/example",
            "require_market_closed": True,
            "require_after_hours_training": False,
        }
    )
    assert result["sent"] is False
    assert result["reason"] == "market_not_closed"


def test_notify_eod_summary_sends_when_training_not_complete_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "health_reason": "market_closed",
        "learning": {},
    }
    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)
    result = slack_srv.tool_notify_eod_summary(
        {
            "webhook_url": "https://hooks.slack.test/example",
            "state_path": str(tmp_path / "slack_eod_state.json"),
            "after_hours_training_marker_path": str(tmp_path / "missing_marker.json"),
        }
    )
    assert result["sent"] is True
    assert len(posts) == 1
    gate = result.get("training_gate")
    assert isinstance(gate, dict)
    assert gate.get("required") is True
    assert gate.get("ready") is False


def test_notify_eod_summary_blocks_until_training_complete_when_configured(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "health_reason": "market_closed",
        "learning": {},
    }

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    result = slack_srv.tool_notify_eod_summary(
        {
            "webhook_url": "https://hooks.slack.test/example",
            "block_on_training_gate": True,
            "after_hours_training_marker_path": str(tmp_path / "missing_marker.json"),
        }
    )
    assert result["sent"] is False
    assert result["reason"] == "after_hours_training_not_complete"
    gate = result.get("training_gate")
    assert isinstance(gate, dict)
    assert gate.get("required") is True
    assert gate.get("ready") is False


def test_notify_eod_summary_allows_send_when_training_marker_matches(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot: dict[str, Any] = {
        "report_date": "2026-03-28",
        "health_reason": "market_closed",
        "learning": {},
    }
    marker_path = tmp_path / "after_hours_training.marker.json"
    marker_path.write_text(
        '{"date":"2026-03-28","status":"trained","model_id":"m1"}\n',
        encoding="utf-8",
    )
    posts: list[dict[str, object]] = []

    def _fake_collect(_args: dict[str, object]) -> dict[str, object]:
        return snapshot

    def _fake_post(webhook_url: str, payload: dict[str, object], timeout_s: float = 5.0) -> int:
        posts.append({"webhook_url": webhook_url, "payload": payload, "timeout_s": timeout_s})
        return 200

    monkeypatch.setattr(slack_srv, "_collect_eod_summary_snapshot", _fake_collect)
    monkeypatch.setattr(slack_srv, "_post_slack_message", _fake_post)

    result = slack_srv.tool_notify_eod_summary(
        {
            "webhook_url": "https://hooks.slack.test/example",
            "state_path": str(tmp_path / "slack_eod_state.json"),
            "after_hours_training_marker_path": str(marker_path),
        }
    )
    assert result["sent"] is True
    gate = result.get("training_gate")
    assert isinstance(gate, dict)
    assert gate.get("ready") is True
    assert len(posts) == 1


def test_extract_report_date_uses_freshest_available_scope() -> None:
    report = {
        "go_no_go": {
            "observed": {
                "trade_metric_scope": {"end_date": "2026-04-24"},
                "gate_metric_scope": {"end_date": "2026-05-01"},
            }
        },
        "execution_vs_alpha": {"daily": [{"date": "2026-04-30"}]},
        "trade_history": {"daily_trade_stats": [{"date": "2026-04-24"}]},
    }

    assert slack_srv._extract_report_date(report) == "2026-05-01"


def test_collect_eod_summary_uses_runtime_file_fallback(monkeypatch) -> None:
    def _fake_runtime_report() -> dict[str, Any]:
        return {
            "go_no_go": {"gate_passed": None, "failed_checks": []},
            "execution_vs_alpha": {"daily": []},
            "trade_history": {"daily_trade_stats": []},
        }

    def _fake_read_json_object(_path: Path) -> dict[str, Any]:
        return {
            "go_no_go": {
                "gate_passed": True,
                "failed_checks": [],
                "observed": {"net_pnl": 10.0},
            },
            "execution_vs_alpha": {
                "execution_capture_ratio": 0.2,
                "slippage_drag_bps": 5.0,
                "daily": [{"date": "2026-03-27"}],
            },
            "trade_history": {"daily_trade_stats": [{"date": "2026-03-27"}]},
        }

    def _fake_health_payload(*, port: int, timeout_s: float) -> dict[str, Any]:
        assert port == 9001
        assert timeout_s == 5.0
        return {
            "status": "healthy",
            "reason": "market_closed",
            "data_provider": {"status": "healthy", "active": "alpaca", "using_backup": False},
            "broker": {"status": "connected"},
            "model_liveness": {},
            "timestamp": "2026-03-29T01:00:00Z",
        }

    monkeypatch.setattr(slack_srv, "_runtime_report_payload", _fake_runtime_report)
    monkeypatch.setattr(slack_srv, "_read_json_object", _fake_read_json_object)
    monkeypatch.setattr(slack_srv, "_health_payload", _fake_health_payload)

    snapshot = slack_srv._collect_eod_summary_snapshot({})
    assert snapshot["report_date"] == "2026-03-27"
    assert snapshot["go_no_go_gate_passed"] is True
    assert snapshot["execution_capture_ratio"] == 0.2


def test_collect_eod_summary_does_not_reuse_stale_trade_row_for_fresh_report_date(
    monkeypatch,
) -> None:
    def _fake_runtime_report() -> dict[str, Any]:
        return {
            "go_no_go": {
                "gate_passed": False,
                "failed_checks": ["win_rate"],
                "observed": {
                    "trade_metric_scope": {"end_date": "2026-04-24"},
                    "gate_metric_scope": {"end_date": "2026-05-01"},
                    "net_pnl": -999.0,
                    "profit_factor": 0.1,
                    "win_rate": 0.2,
                    "closed_trades": 99,
                },
            },
            "execution_vs_alpha": {
                "execution_capture_ratio": 0.2,
                "slippage_drag_bps": 5.0,
                "daily": [{"date": "2026-05-01"}],
            },
            "trade_history": {
                "daily_trade_stats": [
                    {
                        "date": "2026-04-24",
                        "net_pnl": -12.5,
                        "profit_factor": 0.8,
                        "win_rate": 0.4,
                        "trades": 11,
                    }
                ]
            },
        }

    def _fake_health_payload(*, port: int, timeout_s: float) -> dict[str, Any]:
        assert port == 9001
        assert timeout_s == 5.0
        return {
            "status": "healthy",
            "reason": "market_closed",
            "data_provider": {"status": "healthy", "active": "alpaca", "using_backup": False},
            "broker": {"status": "connected"},
            "model_liveness": {},
            "timestamp": "2026-05-02T01:00:00Z",
        }

    monkeypatch.setattr(slack_srv, "_runtime_report_payload", _fake_runtime_report)
    monkeypatch.setattr(slack_srv, "_health_payload", _fake_health_payload)

    snapshot = slack_srv._collect_eod_summary_snapshot({})
    assert snapshot["report_date"] == "2026-05-01"
    assert snapshot["net_pnl"] is None
    assert snapshot["profit_factor"] is None
    assert snapshot["win_rate"] is None
    assert snapshot["closed_trades"] is None
    assert "- Closed trades: n/a" in slack_srv._eod_message_text(snapshot)


def test_collect_eod_summary_prefers_daily_trade_stats_for_daily_kpis(monkeypatch) -> None:
    def _fake_runtime_report() -> dict[str, Any]:
        return {
            "go_no_go": {
                "gate_passed": True,
                "failed_checks": [],
                "observed": {
                    "net_pnl": 999.0,
                    "profit_factor": 9.9,
                    "win_rate": 0.99,
                    "closed_trades": 999,
                },
            },
            "execution_vs_alpha": {
                "execution_capture_ratio": 0.2,
                "slippage_drag_bps": 5.0,
                "daily": [{"date": "2026-03-27"}],
            },
            "trade_history": {
                "daily_trade_stats": [
                    {
                        "date": "2026-03-27",
                        "net_pnl": -12.5,
                        "profit_factor": 0.8,
                        "win_rate": 0.4,
                        "trades": 11,
                    }
                ]
            },
        }

    def _fake_health_payload(*, port: int, timeout_s: float) -> dict[str, Any]:
        assert port == 9001
        assert timeout_s == 5.0
        return {
            "status": "healthy",
            "reason": "market_closed",
            "data_provider": {"status": "healthy", "active": "alpaca", "using_backup": False},
            "broker": {"status": "connected"},
            "model_liveness": {},
            "timestamp": "2026-03-29T01:00:00Z",
        }

    monkeypatch.setattr(slack_srv, "_runtime_report_payload", _fake_runtime_report)
    monkeypatch.setattr(slack_srv, "_health_payload", _fake_health_payload)

    snapshot = slack_srv._collect_eod_summary_snapshot({})
    assert snapshot["report_date"] == "2026-03-27"
    assert snapshot["net_pnl"] == -12.5
    assert snapshot["profit_factor"] == 0.8
    assert snapshot["win_rate"] == 0.4
    assert snapshot["closed_trades"] == 11


def test_collect_eod_summary_uses_nested_report_daily_trade_stats(monkeypatch) -> None:
    def _fake_runtime_report() -> dict[str, Any]:
        return {
            "go_no_go": {
                "gate_passed": False,
                "failed_checks": ["win_rate"],
                "observed": {
                    "net_pnl": 19.88,
                    "profit_factor": 1.18,
                    "win_rate": 0.453,
                    "closed_trades": 97,
                    "trade_metric_scope": {"end_date": "2026-05-04"},
                },
            },
            "report": {
                "execution_vs_alpha": {
                    "execution_capture_ratio": 0.565,
                    "slippage_drag_bps": 8.613,
                    "daily": [{"date": "2026-05-04"}],
                },
                "trade_history": {
                    "daily_trade_stats": [
                        {
                            "date": "2026-05-04",
                            "net_pnl": -0.45,
                            "profit_factor": 0.934,
                            "win_rate": 0.542,
                            "trades": 24,
                        }
                    ],
                    "top_loss_drivers": {
                        "symbols": [{"name": "AMZN", "net_pnl": -3.54}]
                    },
                },
            },
        }

    def _fake_health_payload(*, port: int, timeout_s: float) -> dict[str, Any]:
        assert port == 9001
        assert timeout_s == 5.0
        return {
            "status": "healthy",
            "reason": "market_closed",
            "data_provider": {"status": "healthy", "active": "alpaca", "using_backup": False},
            "broker": {"status": "connected"},
            "model_liveness": {},
            "timestamp": "2026-05-04T20:01:00Z",
        }

    monkeypatch.setattr(slack_srv, "_runtime_report_payload", _fake_runtime_report)
    monkeypatch.setattr(slack_srv, "_health_payload", _fake_health_payload)

    snapshot = slack_srv._collect_eod_summary_snapshot({})
    assert snapshot["report_date"] == "2026-05-04"
    assert snapshot["go_no_go_failed_checks"] == ["win_rate"]
    assert snapshot["net_pnl"] == -0.45
    assert snapshot["profit_factor"] == 0.934
    assert snapshot["win_rate"] == 0.542
    assert snapshot["closed_trades"] == 24
    assert snapshot["top_loss_symbols"] == [{"symbol": "AMZN", "net_pnl": -3.54}]
    message = slack_srv._eod_message_text(snapshot)
    assert "💰 Day performance:" in message
    assert "- Accounting net PnL: $-0.45" in message
    assert "- Closed trades: 24" in message


def test_collect_eod_summary_flags_same_day_fill_pnl_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    fill_events_path = tmp_path / "fill_events.jsonl"
    fill_events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": "2026-05-04T13:45:00+00:00",
                        "symbol": "AMZN",
                        "side": "buy",
                        "fill_qty": 1,
                        "fill_price": 265.41,
                    }
                ),
                json.dumps(
                    {
                        "ts": "2026-05-04T13:47:00+00:00",
                        "symbol": "AMZN",
                        "side": "sell",
                        "fill_qty": 1,
                        "fill_price": 265.22,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_runtime_report() -> dict[str, Any]:
        return {
            "report": {
                "go_no_go": {"gate_passed": False, "failed_checks": ["win_rate"]},
                "execution_vs_alpha": {
                    "execution_capture_ratio": 0.565,
                    "slippage_drag_bps": 8.613,
                    "daily": [{"date": "2026-05-04"}],
                },
                "trade_history": {
                    "daily_trade_stats": [
                        {
                            "date": "2026-05-04",
                            "net_pnl": -7.41,
                            "profit_factor": 0.0,
                            "win_rate": 0.0,
                            "trades": 1,
                        }
                    ],
                },
            },
        }

    def _fake_health_payload(*, port: int, timeout_s: float) -> dict[str, Any]:
        assert port == 9001
        assert timeout_s == 5.0
        return {
            "status": "healthy",
            "reason": "market_closed",
            "data_provider": {"status": "warming_up", "active": "alpaca-iex", "using_backup": False},
            "broker": {"status": "connected"},
            "model_liveness": {},
            "timestamp": "2026-05-04T20:01:00Z",
        }

    monkeypatch.setenv("AI_TRADING_FILL_EVENTS_PATH", str(fill_events_path))
    monkeypatch.setattr(slack_srv, "_runtime_report_payload", _fake_runtime_report)
    monkeypatch.setattr(slack_srv, "_health_payload", _fake_health_payload)

    snapshot = slack_srv._collect_eod_summary_snapshot({})

    assert snapshot["net_pnl"] == -7.41
    assert snapshot["same_day_fill_summary"]["net_pnl"] == -0.18999999999999773
    assert snapshot["same_day_fill_summary"]["closed_trades"] == 1
    assert snapshot["same_day_fill_summary"]["open_qty_by_symbol"] == {}
    assert snapshot["pnl_discrepancy"]["status"] == "mismatch"
    message = slack_srv._eod_message_text(snapshot)
    assert "- Accounting net PnL: $-7.41" in message
    assert "- Same-day fill PnL: $-0.19" in message
    assert "- PnL check: mismatch" in message


def test_collect_eod_summary_degrades_when_health_payload_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        slack_srv,
        "_runtime_report_payload",
        lambda: {
            "go_no_go": {"gate_passed": True, "failed_checks": []},
            "execution_vs_alpha": {
                "execution_capture_ratio": 0.2,
                "daily": [{"date": "2026-03-27"}],
            },
            "trade_history": {"daily_trade_stats": [{"date": "2026-03-27"}]},
        },
    )
    monkeypatch.setattr(
        slack_srv,
        "_health_payload",
        lambda **_: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )

    snapshot = slack_srv._collect_eod_summary_snapshot({})

    assert snapshot["report_date"] == "2026-03-27"
    assert snapshot["health_status"] == "degraded"
    assert snapshot["health_reason"] == "health_payload_unavailable"
    assert snapshot["provider_status"] == "unknown"
    assert snapshot["broker_status"] == "unknown"
    assert isinstance(snapshot["learning"], dict)
