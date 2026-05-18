from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import operator_control_plane


def _health_payload() -> dict[str, object]:
    return {
        "ok": True,
        "status": "ready",
        "attention_flags": ["provider_backup_inactive"],
        "broker": {"connected": True, "open_orders_count": 1, "positions_count": 2},
        "database": {"ok": True},
        "data_provider": {"status": "healthy", "active": "alpaca"},
        "oms_invariants": {"ok": True},
        "oms_lifecycle_parity": {"ok": True},
        "replay_live_parity_gate": {"ok": True},
    }


def test_operator_control_plane_aggregates_read_only_sections(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    report = operator_control_plane.build_operator_control_plane(
        health=_health_payload(),
        readiness={"status": "live_canary_allowed", "reasons": []},
        runtime_gonogo={"gate_passed": True, "failed_checks": []},
        runtime_performance={
            "available": True,
            "source": "runtime_performance_report",
            "go_no_go": {
                "observed": {
                    "open_position_reconciliation_available": True,
                    "open_position_reconciliation_consistent": True,
                    "open_position_reconciliation_mismatch_count": 0,
                }
            },
        },
        oms={"status": "ok"},
        model_registry={"models": {"trend-v2": {"status": "production"}}},
        latest_research={"status": "complete", "candidate_count": 3},
        drift={"status": "ok", "max_psi": 0.04},
        surveillance={"status": "ok"},
        risk_verifier={"status": "pass"},
        paper_sampling={"date": "2026-05-05", "count": 1},
        operator_actions={"actions": [{"id": "review-canary"}]},
        weekend_research={"status": "complete", "monday_preparation": {"status": "complete"}},
        huggingface_research={"status": "discovered", "summary": {"candidate_count": 2}},
        upward_trajectory={"status": "ready", "summary": {"candidate_count": 2}},
    )

    assert report["status"] == "complete"
    assert report["read_only"] is True
    assert report["safety_contract"]["places_orders"] is False
    assert report["safety_contract"]["edits_environment"] is False
    assert report["safety_contract"]["restarts_service"] is False
    assert report["safety_contract"]["patches_code"] is False
    assert report["launch_profile"]["name"] == "live_canary"
    assert report["health"]["ok"] is True
    assert report["go_no_go"]["runtime_gate_passed"] is True
    assert report["orders_positions_oms"]["positions"]["reconciliation_consistent"] is True
    assert report["model_registry"]["artifact_status"] == "present"
    assert report["operator_actions"]["pending_actions"] == [{"id": "review-canary"}]
    assert report["weekend_research"]["artifact_status"] == "complete"
    assert report["weekend_research"]["runtime_authority"] is False
    assert report["huggingface_research"]["artifact_status"] == "discovered"
    assert report["huggingface_research"]["runtime_authority"] is False
    assert report["upward_trajectory"]["artifact_status"] == "ready"
    assert report["upward_trajectory"]["paper_only_diagnostics"] is True
    assert report["upward_trajectory"]["runtime_authority"] is False


def test_operator_control_plane_cli_writes_partial_artifact(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    health = tmp_path / "health.json"
    readiness = tmp_path / "readiness.json"
    output = tmp_path / "operator_control_plane.json"
    health.write_text(json.dumps(_health_payload()), encoding="utf-8")
    readiness.write_text(json.dumps({"status": "paper_only", "reasons": []}), encoding="utf-8")

    exit_code = operator_control_plane.main(
        [
            "--health-json",
            str(health),
            "--readiness-json",
            str(readiness),
            "--runtime-gonogo-json",
            str(tmp_path / "missing_gonogo.json"),
            "--runtime-performance-json",
            str(tmp_path / "missing_runtime.json"),
            "--oms-json",
            str(tmp_path / "missing_oms.json"),
            "--model-registry-json",
            str(tmp_path / "missing_registry.json"),
            "--latest-research-json",
            str(tmp_path / "missing_research.json"),
            "--drift-json",
            str(tmp_path / "missing_drift.json"),
            "--surveillance-json",
            str(tmp_path / "missing_surveillance.json"),
            "--risk-verifier-json",
            str(tmp_path / "missing_risk.json"),
            "--paper-sampling-json",
            str(tmp_path / "missing_sampling.json"),
            "--operator-actions-json",
            str(tmp_path / "missing_actions.json"),
            "--weekend-research-json",
            str(tmp_path / "missing_weekend.json"),
            "--huggingface-research-json",
            str(tmp_path / "missing_hf.json"),
            "--upward-trajectory-json",
            str(tmp_path / "missing_upward.json"),
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "operator_control_plane"
    assert artifact["status"] == "partial"
    assert "runtime_gonogo" in artifact["missing_sections"]
    assert "weekend_research" in artifact["missing_sections"]
    assert "huggingface_research" in artifact["missing_sections"]
    assert "upward_trajectory" in artifact["missing_sections"]
    assert artifact["safety_contract"]["writes"] == ["output_artifact_only"]
