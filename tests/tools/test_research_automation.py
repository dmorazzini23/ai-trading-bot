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
    latest_path = report_root / "latest" / "daily_research_automation_latest.json"
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
    assert "replay_live_cost_alignment" in step_names
    assert "regime_entry_throttle_report" in step_names
    assert "runtime_decay_controls" in step_names
    assert "runtime_gonogo_status" in step_names
    assert "expected_edge_calibration_report" in step_names
    assert "evidence_starvation_report" in step_names
    assert "trading_day_report" in step_names
    assert "symbol_promotion_comparison" in step_names
    assert "daily_research_pipeline" in step_names
    assert "live_capital_readiness" in step_names
    assert "huggingface_research_discovery" in step_names
    assert "huggingface_candidate_intake" in step_names
    assert "huggingface_cache_materialization_plan" in step_names
    assert "evidence_manifest" in payload
    summary = _read(summary_path)
    assert summary["health_report_summary"]["daily_research"]["status"] == "missing"
    assert summary["slack_openclaw_summary"]["service"] == "ai-trading-research-automation"
    readiness = next(step for step in payload["steps"] if step["name"] == "live_capital_readiness")  # type: ignore[index]
    assert readiness["metadata"]["live_money_authority"] is False
    retention = next(step for step in payload["steps"] if step["name"] == "runtime_artifact_retention_plan")  # type: ignore[index]
    assert retention["metadata"]["mutates_runtime_artifacts"] is False
    trading_day = next(step for step in payload["steps"] if step["name"] == "trading_day_report")  # type: ignore[index]
    assert "--report-date" in trading_day["command"]
    assert "--order-intents-jsonl" in trading_day["command"]
    assert "--fills-jsonl" in trading_day["command"]
    assert "--gate-jsonl" in trading_day["command"]
    assert "--regime-entry-throttle-json" in trading_day["command"]
    assert "--expected-edge-calibration-json" in trading_day["command"]
    daily_research = next(step for step in payload["steps"] if step["name"] == "daily_research_pipeline")  # type: ignore[index]
    assert "--symbol-promotion-json" in daily_research["command"]
    assert "--replay-live-cost-alignment-json" in daily_research["command"]
    assert "--regime-entry-throttle-json" in daily_research["command"]
    assert "--training-accelerator-json" in daily_research["command"]
    assert "--expected-edge-calibration-json" in daily_research["command"]
    assert "--evidence-starvation-json" in daily_research["command"]
    assert "--paper-sampling-state-json" in daily_research["command"]
    assert "--huggingface-discovery-json" in daily_research["command"]
    hf_step = next(step for step in payload["steps"] if step["name"] == "huggingface_research_discovery")  # type: ignore[index]
    assert hf_step["metadata"]["runtime_authority"] is False
    assert hf_step["metadata"]["promotion_authority"] is False
    assert hf_step["metadata"]["live_money_authority"] is False
    hf_cache = next(step for step in payload["steps"] if step["name"] == "huggingface_cache_materialization_plan")  # type: ignore[index]
    assert "--dry-run" in hf_cache["command"]


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
    assert "training_accelerator_weekly" in step_names
    assert "microstructure_replay_bridge" in step_names
    assert "huggingface_research_discovery" in step_names
    assert "huggingface_candidate_intake" in step_names
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


def test_manual_promotion_plan_includes_evidence_paths(tmp_path: Path) -> None:
    model = tmp_path / "candidate.joblib"
    model.write_text("model", encoding="utf-8")

    exit_code = research_automation.main(
        [
            "manual",
            "--workflow",
            "promotion",
            "--report-root",
            str(tmp_path / "reports"),
            "--run-id",
            "manual-plan",
            "--model-path",
            str(model),
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = _read(
        tmp_path
        / "reports"
        / "manual"
        / "manual-plan"
        / "research_automation_report.json"
    )
    command = payload["steps"][0]["command"]  # type: ignore[index]
    for flag in (
        "--full-replay-json",
        "--tail-replay-json",
        "--recent-replay-json",
        "--shadow-report-json",
        "--live-cost-model-json",
        "--runtime-decay-controls-json",
    ):
        assert flag in command


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


def test_step_blocked_returncode_is_not_failed(tmp_path: Path) -> None:
    stdout_path = tmp_path / "gonogo.json"
    step = research_automation.ResearchStep(
        name="runtime_gonogo_status",
        command=(
            "bash",
            "-lc",
            "printf '%s\\n' '{\"gate_passed\": false}'; exit 2",
        ),
        purpose="test blocked return code",
        stdout_path=stdout_path,
        blocked_returncodes=(2,),
    )

    result = research_automation._run_step(step)

    assert result["status"] == "blocked"
    assert result["returncode"] == 2
    assert json.loads(stdout_path.read_text(encoding="utf-8")) == {"gate_passed": False}


def test_operator_summary_separates_blocked_from_failed(tmp_path: Path) -> None:
    config = research_automation.ResearchConfig(
        cadence="daily",
        workflow="daily",
        report_root=tmp_path / "reports",
        run_dir=tmp_path / "run",
        run_id="run",
        symbols="AAPL",
        data_dir=None,
        shadow_jsonl=tmp_path / "shadow.jsonl",
        accepted_candidates_jsonl=None,
        model_path=None,
        manifest_path=None,
        current_champion_path="",
        report_date="2026-05-05",
        plan_only=False,
        dry_run=False,
    )

    summary = research_automation._operator_summary(
        config=config,
        status="complete",
        blocked_reasons=[],
        step_results=[
            {"name": "runtime_gonogo_status", "status": "blocked"},
            {"name": "live_cost_model", "status": "passed"},
        ],
        latest_path=tmp_path / "latest.json",
    )

    assert summary["failed_steps"] == []
    assert summary["blocked_steps"] == ["runtime_gonogo_status"]
    assert summary["slack_openclaw_summary"]["suggested_action"] == "review_summary_and_generated_artifacts"


def test_json_stdout_artifact_keeps_final_payload(tmp_path: Path) -> None:
    stdout_path = tmp_path / "status.json"
    step = research_automation.ResearchStep(
        name="runtime_gonogo_status",
        command=(
            "bash",
            "-lc",
            "printf '%s\\n' '{\"msg\":\"startup\"}' '{\"gate_passed\":false,\"failed_checks\":[\"win_rate\"]}'; exit 2",
        ),
        purpose="test stdout normalization",
        stdout_path=stdout_path,
        blocked_returncodes=(2,),
    )

    result = research_automation._run_step(step)
    payload = json.loads(stdout_path.read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert payload == {"failed_checks": ["win_rate"], "gate_passed": False}


def test_run_status_is_blocked_when_any_step_blocks(tmp_path: Path, monkeypatch) -> None:
    config = research_automation.ResearchConfig(
        cadence="daily",
        workflow="daily",
        report_root=tmp_path / "reports",
        run_dir=tmp_path / "run",
        run_id="run",
        symbols="AAPL",
        data_dir=None,
        shadow_jsonl=tmp_path / "shadow.jsonl",
        accepted_candidates_jsonl=None,
        model_path=None,
        manifest_path=None,
        current_champion_path="",
        report_date="2026-05-05",
        plan_only=False,
        dry_run=False,
    )
    step = research_automation.ResearchStep(
        name="runtime_gonogo_status",
        command=("bash", "-lc", "exit 2"),
        purpose="blocked gate",
        blocked_returncodes=(2,),
    )
    monkeypatch.setattr(research_automation, "build_research_steps", lambda _config: ([step], []))

    report = research_automation.run_research_automation(config)

    assert report["status"] == "blocked"
    assert report["step_results"][0]["status"] == "blocked"


def test_authority_artifact_copy_writes_live_readiness_latest(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    run_dir = report_root / "daily" / "run"
    source = run_dir / "live_capital_readiness.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    config = research_automation.ResearchConfig(
        cadence="daily",
        workflow="daily",
        report_root=report_root,
        run_dir=run_dir,
        run_id="run",
        symbols="AAPL",
        data_dir=None,
        shadow_jsonl=tmp_path / "shadow.jsonl",
        accepted_candidates_jsonl=None,
        model_path=None,
        manifest_path=None,
        current_champion_path="",
        report_date="2026-05-05",
        plan_only=False,
        dry_run=False,
    )

    copied = research_automation._copy_authority_artifacts(
        config=config,
        step_results=[
            {
                "name": "live_capital_readiness",
                "status": "blocked",
                "output_path": str(source),
            }
        ],
    )

    assert "live_capital_readiness" in copied
    assert (report_root / "latest" / "live_capital_readiness_latest.json").is_file()


def test_authority_artifact_copy_writes_daily_evidence_latest_paths(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    run_dir = report_root / "daily" / "run"
    run_dir.mkdir(parents=True)
    sources = {
        "live_cost_model": run_dir / "live_cost_model.json",
        "ml_shadow_report": run_dir / "ml_shadow_report.json",
        "replay_governance_refresh": run_dir / "replay_governance_summary.json",
        "daily_research_pipeline": run_dir / "daily_research_report.json",
    }
    for name, source in sources.items():
        source.write_text(json.dumps({"artifact": name}), encoding="utf-8")
    config = research_automation.ResearchConfig(
        cadence="daily",
        workflow="daily",
        report_root=report_root,
        run_dir=run_dir,
        run_id="run",
        symbols="AAPL",
        data_dir=None,
        shadow_jsonl=tmp_path / "shadow.jsonl",
        accepted_candidates_jsonl=None,
        model_path=None,
        manifest_path=None,
        current_champion_path="",
        report_date="2026-05-05",
        plan_only=False,
        dry_run=False,
    )

    copied = research_automation._copy_authority_artifacts(
        config=config,
        step_results=[
            {"name": name, "status": "passed", "output_path": str(path)}
            for name, path in sources.items()
        ],
    )

    assert set(sources).issubset(copied)
    assert json.loads((report_root / "latest" / "live_cost_model_latest.json").read_text()) == {
        "artifact": "live_cost_model"
    }
    assert json.loads((report_root / "latest" / "ml_shadow_report_latest.json").read_text()) == {
        "artifact": "ml_shadow_report"
    }
    assert json.loads((report_root / "latest" / "replay_governance_refresh_latest.json").read_text()) == {
        "artifact": "replay_governance_refresh"
    }
    assert json.loads((report_root / "latest" / "daily_research_latest.json").read_text()) == {
        "artifact": "daily_research_pipeline"
    }


def test_operator_summary_reads_next_level_latest_artifacts(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    latest = report_root / "latest"
    latest.mkdir(parents=True)
    (latest / "daily_readiness_latest.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "trade_allowed": False,
                "recommended_next_session_mode": "observe",
                "blocked_reasons": ["runtime_gonogo_failed"],
            }
        ),
        encoding="utf-8",
    )
    (latest / "trading_day_latest.json").write_text(
        json.dumps(
            {
                "desired_trades": {"count": 3},
                "submitted_trades": {"count": 2},
                "rejected_trades": {"count": 1},
                "realized_fills": {"count": 2},
            }
        ),
        encoding="utf-8",
    )
    (latest / "live_capital_readiness_latest.json").write_text(
        json.dumps({"status": "blocked", "reasons": ["live_account_not_explicitly_confirmed"]}),
        encoding="utf-8",
    )
    config = research_automation.ResearchConfig(
        cadence="daily",
        workflow="daily",
        report_root=report_root,
        run_dir=tmp_path / "run",
        run_id="run",
        symbols="AAPL",
        data_dir=None,
        shadow_jsonl=tmp_path / "shadow.jsonl",
        accepted_candidates_jsonl=None,
        model_path=None,
        manifest_path=None,
        current_champion_path="",
        report_date="2026-05-05",
        plan_only=False,
        dry_run=False,
    )

    summary = research_automation._operator_summary(
        config=config,
        status="blocked",
        blocked_reasons=[],
        step_results=[],
        latest_path=latest / "daily_research_automation_latest.json",
    )

    assert summary["health_report_summary"]["daily_research"]["trade_allowed"] is False
    assert summary["health_report_summary"]["trading_day"]["fills"] == 2
    assert summary["health_report_summary"]["live_capital_readiness"]["status"] == "blocked"
    assert summary["slack_openclaw_summary"]["next_level_artifacts"]["daily_research"]["blocked_reasons"] == [
        "runtime_gonogo_failed"
    ]


def test_run_research_automation_keeps_daily_report_latest_for_daily_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    config = research_automation.ResearchConfig(
        cadence="daily",
        workflow="daily",
        report_root=tmp_path / "reports",
        run_dir=tmp_path / "run",
        run_id="run",
        symbols="AAPL",
        data_dir=None,
        shadow_jsonl=tmp_path / "shadow.jsonl",
        accepted_candidates_jsonl=None,
        model_path=None,
        manifest_path=None,
        current_champion_path="",
        report_date="2026-05-05",
        plan_only=False,
        dry_run=False,
    )
    daily_source = config.run_dir / "daily_research_report.json"
    step = research_automation.ResearchStep(
        name="daily_research_pipeline",
        command=("bash", "-lc", f"printf '{{\"trade_allowed\":false}}' > {daily_source}"),
        purpose="write daily report",
        output_path=daily_source,
    )
    monkeypatch.setattr(research_automation, "build_research_steps", lambda _config: ([step], []))

    report = research_automation.run_research_automation(config)

    assert report["status"] == "complete"
    assert json.loads(
        (config.report_root / "latest" / "daily_research_latest.json").read_text()
    ) == {"trade_allowed": False}
    automation_latest = json.loads(
        (config.report_root / "latest" / "daily_research_automation_latest.json").read_text()
    )
    assert automation_latest["artifact_type"] == "research_automation_report"


def test_blocked_research_step_surfaces_operator_reason(tmp_path: Path, monkeypatch) -> None:
    config = research_automation.ResearchConfig(
        cadence="daily",
        workflow="daily",
        report_root=tmp_path / "reports",
        run_dir=tmp_path / "run",
        run_id="run",
        symbols="AAPL,AMZN",
        data_dir=None,
        shadow_jsonl=tmp_path / "shadow.jsonl",
        accepted_candidates_jsonl=None,
        model_path=None,
        manifest_path=None,
        current_champion_path="",
        report_date="2026-05-05",
        plan_only=False,
        dry_run=False,
    )
    output = config.run_dir / "replay_governance_summary.json"
    step = research_automation.ResearchStep(
        name="replay_governance_refresh",
        command=(
            "bash",
            "-lc",
            (
                f"printf '%s\\n' '{{\"status\":\"blocked\","
                f"\"reason\":\"REPLAY_POLICY_NON_REGRESSION_FAILED\"}}' > {output}; exit 2"
            ),
        ),
        purpose="blocked replay governance",
        output_path=output,
        blocked_returncodes=(2,),
    )
    monkeypatch.setattr(research_automation, "build_research_steps", lambda _config: ([step], []))

    report = research_automation.run_research_automation(config)
    summary = json.loads(
        (config.report_root / "latest" / "daily_operator_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "blocked"
    assert summary["blocked_steps"] == ["replay_governance_refresh"]
    assert summary["blocked_reasons"] == [
        "replay_governance_refresh:REPLAY_POLICY_NON_REGRESSION_FAILED"
    ]
