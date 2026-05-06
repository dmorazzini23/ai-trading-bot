from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_trading.tools import research_automation


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_daily_plan_writes_artifacts_without_running_steps(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    exit_code = research_automation.main(
        [
            "daily",
            "--report-root",
            str(report_root),
            "--run-id",
            "daily-test",
            "--symbols",
            "AAPL,AMZN",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    report_path = report_root / "daily" / "daily-test" / "research_automation_report.json"
    latest_path = report_root / "latest" / "daily_research_latest.json"
    summary_path = report_root / "latest" / "daily_operator_summary.json"
    assert report_path.is_file()
    assert latest_path.is_file()
    assert summary_path.is_file()
    payload = _read(report_path)
    assert payload["artifact_type"] == "research_automation_report"
    assert payload["status"] == "planned"
    assert payload["safety"] == {
        "automated_runtime_mutations": False,
        "live_money_cutover": "manual_only",
        "production_model_promotion": "manual_only",
        "slack_openclaw_source": "generated_artifacts",
    }
    step_names = {str(step["name"]) for step in payload["steps"]}  # type: ignore[index]
    assert "memory_hotspot_audit" in step_names
    assert "runtime_artifact_retention_plan" in step_names
    assert "live_cost_model" in step_names
    assert "runtime_decay_controls" in step_names
    assert "runtime_gonogo_status" in step_names
    assert "trading_day_report" in step_names
    assert "daily_research_pipeline" in step_names
    assert "live_capital_readiness" in step_names
    readiness = next(step for step in payload["steps"] if step["name"] == "live_capital_readiness")  # type: ignore[index]
    assert readiness["metadata"]["live_money_authority"] is False
    retention = next(step for step in payload["steps"] if step["name"] == "runtime_artifact_retention_plan")  # type: ignore[index]
    assert retention["metadata"]["mutates_runtime_artifacts"] is False


def test_weekly_plan_adds_multi_horizon_and_microstructure_when_inputs_exist(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "bars"
    data_dir.mkdir()
    shadow = tmp_path / "shadow.jsonl"
    accepted = tmp_path / "accepted_candidates.jsonl"
    shadow.write_text("{}", encoding="utf-8")
    accepted.write_text("{}", encoding="utf-8")

    exit_code = research_automation.main(
        [
            "weekly",
            "--report-root",
            str(tmp_path / "reports"),
            "--run-id",
            "weekly-test",
            "--data-dir",
            str(data_dir),
            "--shadow-jsonl",
            str(shadow),
            "--accepted-candidates-jsonl",
            str(accepted),
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = _read(
        tmp_path
        / "reports"
        / "weekly"
        / "weekly-test"
        / "research_automation_report.json"
    )
    step_names = {str(step["name"]) for step in payload["steps"]}  # type: ignore[index]
    assert "multi_horizon_objective_search" in step_names
    assert "microstructure_replay_bridge" in step_names
    bridge = next(step for step in payload["steps"] if step["name"] == "microstructure_replay_bridge")  # type: ignore[index]
    assert bridge["metadata"]["enforcement_authority"] is False


def test_manual_promotion_blocks_without_model_path(tmp_path: Path) -> None:
    exit_code = research_automation.main(
        [
            "manual",
            "--workflow",
            "promotion",
            "--report-root",
            str(tmp_path / "reports"),
            "--run-id",
            "manual-test",
        ]
    )

    assert exit_code == 2
    payload = _read(
        tmp_path
        / "reports"
        / "manual"
        / "manual-test"
        / "research_automation_report.json"
    )
    assert payload["status"] == "blocked"
    assert payload["blocked_reasons"] == ["manual_promotion_requires_model_path"]
    summary = _read(tmp_path / "reports" / "latest" / "manual_operator_summary.json")
    assert summary["operator_action"] == "resolve_blocked_reasons_then_rerun"


def test_manual_live_cutover_plan_has_no_live_money_authority(tmp_path: Path) -> None:
    exit_code = research_automation.main(
        [
            "manual",
            "--workflow",
            "live-cutover",
            "--report-root",
            str(tmp_path / "reports"),
            "--run-id",
            "cutover-test",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = _read(
        tmp_path
        / "reports"
        / "manual"
        / "cutover-test"
        / "research_automation_report.json"
    )
    step = payload["steps"][0]  # type: ignore[index]
    assert step["name"] == "manual_live_cutover_drill"
    assert step["metadata"]["live_money_authority"] is False
    assert payload["safety"]["live_money_cutover"] == "manual_only"
