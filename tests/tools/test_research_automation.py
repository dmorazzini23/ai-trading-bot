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
    assert "upward_trajectory_report" in step_names
    assert "evidence_manifest" in payload
    calibration = next(
        step
        for step in payload["steps"]  # type: ignore[index]
        if step["name"] == "expected_edge_calibration_report"
    )
    calibration_command = [str(token) for token in calibration["command"]]
    assert calibration_command[calibration_command.index("--tca-jsonl") + 1].endswith(
        "/runtime/tca_records.jsonl"
    )
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
    assert "--decisions-jsonl" in trading_day["command"]
    decisions_index = trading_day["command"].index("--decisions-jsonl")
    assert trading_day["command"][decisions_index + 1].endswith(
        "/runtime/decision_records.jsonl"
    )
    assert "--regime-entry-throttle-json" in trading_day["command"]
    assert "--expected-edge-calibration-json" in trading_day["command"]
    enriched_trading_day = next(
        step for step in payload["steps"] if step["name"] == "trading_day_report_enriched"
    )  # type: ignore[index]
    assert "--weekend-research-json" in enriched_trading_day["command"]
    assert "--decisions-jsonl" in enriched_trading_day["command"]
    enriched_decisions_index = enriched_trading_day["command"].index(
        "--decisions-jsonl"
    )
    assert enriched_trading_day["command"][enriched_decisions_index + 1].endswith(
        "/runtime/decision_records.jsonl"
    )
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
    assert hf_step["metadata"]["non_authoritative"] is True
    assert hf_step["metadata"]["requires_explicit_api_opt_in"] is True
    hf_cache = next(step for step in payload["steps"] if step["name"] == "huggingface_cache_materialization_plan")  # type: ignore[index]
    assert "--dry-run" in hf_cache["command"]
    registry = next(step for step in payload["steps"] if step["name"] == "model_registry_evaluation")  # type: ignore[index]
    registry_command = [str(token) for token in registry["command"]]
    registry_json_arg = registry_command[registry_command.index("--registry-json") + 1]
    assert registry_json_arg.endswith("/models/registry_index.json")
    assert not registry_json_arg.endswith("/research_reports/latest/model_registry_latest.json")
    ordered_step_names = [str(step["name"]) for step in payload["steps"]]  # type: ignore[index]
    current_drift = next(
        step
        for step in payload["steps"]  # type: ignore[index]
        if step["name"] == "model_data_drift_current_evidence"
    )
    drift_monitor = next(
        step
        for step in payload["steps"]  # type: ignore[index]
        if step["name"] == "model_data_drift_monitor"
    )
    assert ordered_step_names.index("model_registry_evaluation") < ordered_step_names.index(
        "model_data_drift_current_evidence"
    )
    assert ordered_step_names.index(
        "model_data_drift_current_evidence"
    ) < ordered_step_names.index("model_data_drift_monitor")
    current_command = [str(token) for token in current_drift["command"]]
    monitor_command = [str(token) for token in drift_monitor["command"]]
    registry_output = str(registry["output_path"])
    current_output = str(current_drift["output_path"])
    assert current_command[current_command.index("--model-registry-json") + 1] == registry_output
    assert current_command[current_command.index("--fills-jsonl") + 1].endswith(
        "/runtime/fill_events.jsonl"
    )
    assert current_command[current_command.index("--tca-jsonl") + 1].endswith(
        "/runtime/tca_records.jsonl"
    )
    assert monitor_command[monitor_command.index("--current-json") + 1] == current_output
    assert "expected_edge_calibration.json" not in monitor_command
    assert current_drift["metadata"]["baseline_mutation"] is False
    assert drift_monitor["metadata"]["baseline_mutation"] is False


def test_hf_research_api_requires_second_explicit_toggle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_HF_RESEARCH_ENABLED", "1")
    monkeypatch.delenv("AI_TRADING_HF_RESEARCH_USE_API", raising=False)

    exit_code = research_automation.main(
        [
            "daily",
            "--report-root",
            str(tmp_path / "reports"),
            "--run-id",
            "hf-toggle-test",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = _read(
        tmp_path
        / "reports"
        / "daily"
        / "hf-toggle-test"
        / "research_automation_report.json"
    )
    hf_step = next(step for step in payload["steps"] if step["name"] == "huggingface_research_discovery")  # type: ignore[index]
    assert "--enabled" in hf_step["command"]
    assert "--use-hf-api" not in hf_step["command"]

    monkeypatch.setenv("AI_TRADING_HF_RESEARCH_USE_API", "1")
    exit_code = research_automation.main(
        [
            "daily",
            "--report-root",
            str(tmp_path / "reports-api"),
            "--run-id",
            "hf-toggle-test",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = _read(
        tmp_path
        / "reports-api"
        / "daily"
        / "hf-toggle-test"
        / "research_automation_report.json"
    )
    hf_step = next(step for step in payload["steps"] if step["name"] == "huggingface_research_discovery")  # type: ignore[index]
    assert "--enabled" in hf_step["command"]
    assert "--use-hf-api" in hf_step["command"]


def test_daily_plan_with_data_adds_upward_trajectory_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    data_dir.mkdir()
    report_root = tmp_path / "reports"

    exit_code = research_automation.main(
        [
            "daily",
            "--report-root",
            str(report_root),
            "--run-id",
            "upward-test",
            "--data-dir",
            str(data_dir),
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = _read(report_root / "daily" / "upward-test" / "research_automation_report.json")
    step_names = [str(step["name"]) for step in payload["steps"]]  # type: ignore[index]
    assert "upward_trajectory_report" in step_names
    assert step_names.index("regime_champion_models") < step_names.index("upward_trajectory_report")
    assert step_names.index("upward_trajectory_report") < step_names.index("operator_control_plane")
    upward = next(step for step in payload["steps"] if step["name"] == "upward_trajectory_report")  # type: ignore[index]
    assert upward["metadata"]["research_only"] is True
    assert upward["metadata"]["live_money_authority"] is False
    assert "--training-accelerator-json" in upward["command"]
    daily_research = next(step for step in payload["steps"] if step["name"] == "daily_research_pipeline")  # type: ignore[index]
    assert "--upward-trajectory-json" in daily_research["command"]


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


def test_weekend_saturday_plan_uses_bounded_broad_research_caps(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "bars"
    data_dir.mkdir()
    report_root = tmp_path / "reports"
    exit_code = research_automation.main(
        [
            "weekend-saturday",
            "--report-root",
            str(report_root),
            "--run-id",
            "saturday-test",
            "--data-dir",
            str(data_dir),
            "--symbols",
            "AAPL,AMZN,MSFT,NVDA",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = _read(report_root / "weekend" / "saturday-test" / "weekend_research_report.json")
    assert payload["status"] == "planned"
    assert payload["cadence"] == "weekend-saturday"
    assert payload["weekend_schedule"]["max_runtime_minutes"] == 180
    assert payload["weekend_schedule"]["max_symbols"] == 25
    assert payload["weekend_schedule"]["max_candidates"] == 100
    assert payload["weekend_schedule"]["max_replay_candidates"] == 15
    assert payload["monday_preparation"]["research_only"] is True
    assert payload["safety"]["weekend_research_authority"] == "research_only"
    step_names = {str(step["name"]) for step in payload["steps"]}  # type: ignore[index]
    assert "training_accelerator_weekend_broad" in step_names
    assert "multi_horizon_weekend_broad" in step_names
    broad = next(step for step in payload["steps"] if step["name"] == "multi_horizon_weekend_broad")  # type: ignore[index]
    assert "--max-replay-candidates" in broad["command"]
    assert "15" in broad["command"]
    assert broad["metadata"]["promotion_authority"] is False
    assert broad["metadata"]["manual_approval_required"] is True
    assert (report_root / "latest" / "weekend_research_latest.json").is_file()
    assert (report_root / "latest" / "weekend_operator_summary.json").is_file()


def test_weekend_sunday_plan_builds_monday_readiness_synthesis(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    exit_code = research_automation.main(
        [
            "weekend-sunday",
            "--report-root",
            str(report_root),
            "--run-id",
            "sunday-test",
            "--plan-only",
        ]
    )

    assert exit_code == 0
    payload = _read(report_root / "weekend" / "sunday-test" / "weekend_research_report.json")
    assert payload["status"] == "planned"
    assert payload["cadence"] == "weekend-sunday"
    assert payload["weekend_schedule"]["max_runtime_minutes"] == 120
    assert payload["weekend_schedule"]["max_replay_candidates"] == 20
    assert payload["monday_preparation"]["recommended_operator_action"] == (
        "review_monday_preparation_before_market_open"
    )
    step_names = {str(step["name"]) for step in payload["steps"]}  # type: ignore[index]
    for expected in (
        "replay_governance_refresh",
        "replay_live_cost_alignment",
        "walk_forward_capital_simulation",
        "counterfactual_execution_replay",
        "regime_entry_throttle_report",
        "order_type_optimizer",
        "post_trade_surveillance",
        "model_data_drift_monitor",
        "operator_control_plane",
        "live_capital_readiness",
    ):
        assert expected in step_names
    readiness = next(step for step in payload["steps"] if step["name"] == "live_capital_readiness")  # type: ignore[index]
    assert readiness["metadata"]["live_money_authority"] is False
    assert readiness["metadata"]["manual_approval_required"] is True


def test_weekend_research_disabled_blocks_without_steps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        research_automation,
        "_env_text",
        lambda name, default: "0"
        if name == "AI_TRADING_WEEKEND_RESEARCH_ENABLED"
        else default,
    )

    exit_code = research_automation.main(
        [
            "weekend-saturday",
            "--report-root",
            str(tmp_path / "reports"),
            "--run-id",
            "disabled-test",
        ]
    )

    assert exit_code == 2
    payload = _read(
        tmp_path / "reports" / "weekend" / "disabled-test" / "weekend_research_report.json"
    )
    assert payload["status"] == "blocked"
    assert payload["blocked_reasons"] == ["weekend_research_disabled"]
    assert payload["steps"] == []


def test_weekend_cap_exhaustion_marks_remaining_steps_failed(
    tmp_path: Path, monkeypatch
) -> None:
    config = research_automation.ResearchConfig(
        cadence="weekend-saturday",
        workflow="weekend-saturday",
        report_root=tmp_path / "reports",
        run_dir=tmp_path / "reports" / "weekend" / "run",
        run_id="run",
        symbols="AAPL",
        data_dir=None,
        shadow_jsonl=tmp_path / "shadow.jsonl",
        accepted_candidates_jsonl=None,
        model_path=None,
        manifest_path=None,
        current_champion_path="",
        report_date="2026-05-09",
        plan_only=False,
        dry_run=False,
    )
    step = research_automation.ResearchStep(
        name="slow_weekend_step",
        command=("bash", "-lc", "sleep 2"),
        purpose="exercise timeout",
    )
    monkeypatch.setattr(research_automation, "build_research_steps", lambda _config: ([step], []))
    monkeypatch.setattr(research_automation, "_weekend_cap_summary", lambda _config: {
        "enabled": True,
        "max_runtime_minutes": 15,
        "max_symbols": 1,
        "max_candidates": 1,
        "max_replay_candidates": 1,
        "max_parallel_workers": 1,
        "cache_enabled": True,
        "effective_symbols": "AAPL",
    })
    times = iter([0.0, 901.0])
    monkeypatch.setattr(research_automation, "monotonic", lambda: next(times, 901.0))

    report = research_automation.run_research_automation(config)

    assert report["status"] == "failed"
    assert report["step_results"][0]["reason"] == "weekend_runtime_cap_exhausted"


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
    assert summary["required_blocked_steps"] == []
    assert summary["optional_blocked_steps"] == ["runtime_gonogo_status"]
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


def test_run_status_is_complete_when_optional_step_blocks(tmp_path: Path, monkeypatch) -> None:
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

    assert report["status"] == "complete"
    assert report["step_results"][0]["status"] == "blocked"


def test_run_status_is_blocked_when_required_step_blocks(tmp_path: Path, monkeypatch) -> None:
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
        name="live_capital_readiness",
        command=("bash", "-lc", "exit 2"),
        purpose="required blocked gate",
        required=True,
        blocked_returncodes=(2,),
    )
    monkeypatch.setattr(research_automation, "build_research_steps", lambda _config: ([step], []))

    report = research_automation.run_research_automation(config)

    assert report["status"] == "blocked"
    assert report["step_results"][0]["status"] == "blocked"


def test_blocked_authority_artifact_copy_does_not_seed_live_readiness_latest(tmp_path: Path) -> None:
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

    assert "live_capital_readiness" not in copied
    assert not (report_root / "latest" / "live_capital_readiness_latest.json").exists()


def test_blocked_authority_artifact_does_not_overwrite_existing_latest(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    run_dir = report_root / "daily" / "run"
    source = run_dir / "live_capital_readiness.json"
    latest = report_root / "latest" / "live_capital_readiness_latest.json"
    source.parent.mkdir(parents=True)
    latest.parent.mkdir(parents=True)
    source.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    latest.write_text(json.dumps({"status": "live_canary_allowed"}), encoding="utf-8")
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

    research_automation._copy_authority_artifacts(
        config=config,
        step_results=[
            {
                "name": "live_capital_readiness",
                "status": "blocked",
                "output_path": str(source),
            }
        ],
    )

    assert json.loads(latest.read_text(encoding="utf-8")) == {"status": "live_canary_allowed"}


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
    runtime_root = tmp_path / "runtime"
    stale_research_latest = config.report_root / "latest" / "replay_governance_refresh_latest.json"
    stale_runtime_latest = runtime_root / "runtime/replay_governance_refresh_latest.json"
    stale_research_latest.parent.mkdir(parents=True, exist_ok=True)
    stale_runtime_latest.parent.mkdir(parents=True, exist_ok=True)
    stale_payload = {"status": "ok", "reason": None}
    stale_research_latest.write_text(json.dumps(stale_payload), encoding="utf-8")
    stale_runtime_latest.write_text(json.dumps(stale_payload), encoding="utf-8")

    def fake_runtime_artifact_path(
        path_value: str,
        *,
        default_relative: str | None = None,
        for_write: bool = False,
    ) -> Path:
        del for_write
        return runtime_root / (default_relative or path_value)

    monkeypatch.setattr(
        research_automation,
        "resolve_runtime_artifact_path",
        fake_runtime_artifact_path,
    )
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
        required=True,
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
    assert report["paths"]["authority_copies"]["replay_governance_refresh"] == str(
        runtime_root / "runtime/replay_governance_refresh_latest.json"
    )
    assert json.loads(
        (config.report_root / "latest" / "replay_governance_refresh_latest.json").read_text(
            encoding="utf-8"
        )
    ) == {
        "status": "blocked",
        "reason": "REPLAY_POLICY_NON_REGRESSION_FAILED",
    }
    assert json.loads(
        (runtime_root / "runtime/replay_governance_refresh_latest.json").read_text(
            encoding="utf-8"
        )
    ) == {
        "status": "blocked",
        "reason": "REPLAY_POLICY_NON_REGRESSION_FAILED",
    }
    assert summary["blocked_steps"] == ["replay_governance_refresh"]
    assert summary["blocked_reasons"] == [
        "replay_governance_refresh:REPLAY_POLICY_NON_REGRESSION_FAILED"
    ]
