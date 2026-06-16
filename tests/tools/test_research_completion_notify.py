from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ai_trading.tools import research_completion_notify


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload)), encoding="utf-8")


def test_research_completion_payload_reads_latest_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_json(
        root / "latest" / "daily_research_automation_latest.json",
        {
            "status": "complete",
            "config": {"run_id": "daily-20260505T210000Z", "run_dir": "/runtime/research/daily/run"},
            "paths": {"report": "/runtime/research/research_automation_report.json"},
            "steps": [
                {
                    "name": "live_capital_readiness",
                    "output_path": str(root / "daily" / "run" / "live_capital_readiness.json"),
                }
            ],
            "step_results": [
                {"name": "live_cost_model", "status": "passed"},
                {"name": "runtime_gonogo_status", "status": "blocked"},
                {"name": "multi_horizon_lightweight", "status": "skipped"},
            ],
        },
    )
    _write_json(
        root / "latest" / "daily_operator_summary.json",
        {
            "operator_action": "review_summary_and_generated_artifacts",
            "blocked_reasons": [],
            "health_report_summary": {
                "daily_research": {"trade_allowed": True},
                "trading_day": {"fills": 2},
            },
            "slack_openclaw_summary": {
                "summary": "research_automation cadence=daily workflow=daily status=complete"
            },
        },
    )
    _write_json(
        root / "latest" / "daily_readiness_latest.json",
        {"recommended_next_session_mode": "paper_trade", "trade_allowed": True},
    )
    _write_json(
        root / "latest" / "trading_day_latest.json",
        {
            "desired_trades": {"count": 3},
            "submitted_trades": {"count": 2},
            "rejected_trades": {"count": 1},
            "controlled_skips": {"count": 4},
            "realized_fills": {"count": 2},
        },
    )
    _write_json(
        root / "latest" / "symbol_promotion_latest.json",
        {
            "promotion_authority": False,
            "runtime_symbol_gating_changed": False,
            "symbols": [
                {
                    "symbol": "AMZN",
                    "recommendation": "consider_promotion",
                    "confidence": "high",
                },
                {
                    "symbol": "MSFT",
                    "recommendation": "collect_more_evidence",
                    "confidence": "low",
                },
            ],
        },
    )
    _write_json(
        root / "latest" / "expected_edge_calibration_latest.json",
        {"status": "overestimated", "recommended_next_action": "keep_tiny_sampling"},
    )
    _write_json(
        root / "latest" / "evidence_starvation_latest.json",
        {"status": "starved", "recommendation": "widen_paper_diagnostic_sampling"},
    )
    _write_json(
        root / "latest" / "hf_discovery_latest.json",
        {"status": "discovered", "summary": {"candidate_count": 4, "accepted_for_offline_experiment": 1}},
    )
    _write_json(
        root / "latest" / "hf_candidate_intake_latest.json",
        {"status": "ready_for_manual_review", "summary": {"accepted_for_offline_experiment": 1}},
    )
    _write_json(root / "daily" / "run" / "live_capital_readiness.json", {"status": "blocked"})

    payload = research_completion_notify.build_research_completion_payload(
        cadence="daily",
        workflow="daily",
        exit_code=0,
        report_root=root,
        channel="#all-beatwallstreet",
    )

    assert payload["channel"] == "#all-beatwallstreet"
    assert payload["text"].startswith("ai-trading research daily finished: complete")
    section_field_counts = [
        len(block.get("fields", []))
        for block in payload["blocks"]
        if block.get("type") == "section"
    ]
    assert section_field_counts
    assert max(section_field_counts) <= 10
    field_text = "\n".join(
        field["text"]
        for block in payload["blocks"]
        if block.get("type") == "section"
        for field in block.get("fields", [])
    )
    assert "paper_trade" in field_text
    assert "daily:daily:daily-20260505T210000Z" in field_text
    assert "/runtime/research/daily/run" in field_text
    assert "blocked" in field_text
    assert "multi_horizon_lightweight" in field_text
    assert "runtime_gonogo_status" in field_text
    assert "desired=3" in field_text
    assert "controlled_skips=4" in field_text
    assert "AMZN:consider_promotion/high" in field_text
    assert "MSFT:collect_more_evidence/low" in field_text
    assert "overestimated / keep_tiny_sampling" in field_text
    assert "starved / widen_paper_diagnostic_sampling" in field_text
    assert "discovered / scanned=4, accepted=1, runtime_authority=false" in field_text
    assert "research_automation cadence=daily workflow=daily status=complete" in field_text
    assert '"trade_allowed": true' in field_text


def test_research_completion_payload_includes_weekend_artifact_summary(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_json(
        root / "latest" / "weekend-saturday_research_automation_latest.json",
        {
            "status": "complete",
            "cadence": "weekend-saturday",
            "weekend_schedule": {
                "effective_symbols": "AAPL,AMZN",
                "max_candidates": 100,
                "max_replay_candidates": 15,
            },
            "monday_preparation": {
                "recommended_operator_action": "review_broad_research_then_wait_for_sunday_validation"
            },
            "paths": {"report": "/runtime/research/weekend_research_report.json"},
            "step_results": [
                {"name": "training_accelerator_weekend_broad", "status": "passed"},
            ],
        },
    )
    _write_json(
        root / "latest" / "weekend-saturday_operator_summary.json",
        {
            "operator_action": "review_broad_research_then_wait_for_sunday_validation",
            "blocked_reasons": [],
        },
    )

    payload = research_completion_notify.build_research_completion_payload(
        cadence="weekend-saturday",
        workflow="weekend-saturday",
        exit_code=0,
        report_root=root,
        channel="#all-beatwallstreet",
    )

    field_text = "\n".join(
        field["text"]
        for block in payload["blocks"]
        if block.get("type") == "section"
        for field in block.get("fields", [])
    )
    assert "symbols=AAPL,AMZN" in field_text
    assert "max_candidates=100" in field_text
    assert "max_replay=15" in field_text
    assert "research_only=true" in field_text
    assert "review_broad_research_then_wait_for_sunday_validation" in field_text


def test_research_completion_payload_marks_failed_exit_code_failed(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_json(
        root / "latest" / "daily_research_automation_latest.json",
        {
            "status": "complete",
            "paths": {"report": "/runtime/research/stale_complete.json"},
        },
    )
    _write_json(
        root / "latest" / "daily_operator_summary.json",
        {"operator_action": "review_prior_report"},
    )

    payload = research_completion_notify.build_research_completion_payload(
        cadence="daily",
        workflow="daily",
        exit_code=1,
        report_root=root,
        channel="#all-beatwallstreet",
        run_status="infrastructure_failed",
    )

    assert payload["text"].startswith("ai-trading research daily finished: infrastructure_failed")
    field_text = "\n".join(
        field["text"]
        for block in payload["blocks"]
        if block.get("type") == "section"
        for field in block.get("fields", [])
    )
    assert "*Report status*\nunknown" in field_text
    assert "*Artifact freshness*\nstale_latest_suppressed" in field_text
    assert "stale_complete.json" not in field_text


def test_research_completion_payload_preserves_blocked_status_on_blocked_exit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports"
    _write_json(
        root / "latest" / "daily_research_automation_latest.json",
        {
            "status": "blocked",
            "paths": {"report": "/runtime/research/blocked.json"},
            "step_results": [{"name": "runtime_gonogo_status", "status": "blocked"}],
        },
    )
    _write_json(
        root / "latest" / "daily_operator_summary.json",
        {"operator_action": "resolve_blocked_reasons_then_rerun"},
    )

    payload = research_completion_notify.build_research_completion_payload(
        cadence="daily",
        workflow="daily",
        exit_code=2,
        report_root=root,
        channel="#all-beatwallstreet",
    )

    assert payload["text"].startswith("ai-trading research daily finished: blocked")


def test_research_completion_payload_suppresses_locked_stale_latest(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    _write_json(
        root / "latest" / "daily_research_automation_latest.json",
        {
            "status": "complete",
            "paths": {"report": "/runtime/research/old_complete.json"},
        },
    )
    _write_json(
        root / "latest" / "daily_operator_summary.json",
        {"operator_action": "review_prior_report"},
    )

    payload = research_completion_notify.build_research_completion_payload(
        cadence="daily",
        workflow="daily",
        exit_code=75,
        report_root=root,
        channel="#all-beatwallstreet",
        run_status="locked",
    )

    assert payload["text"].startswith("ai-trading research daily finished: locked")
    field_text = "\n".join(
        field["text"]
        for block in payload["blocks"]
        if block.get("type") == "section"
        for field in block.get("fields", [])
    )
    assert "*Report status*\nunknown" in field_text
    assert "*Artifact freshness*\nstale_latest_suppressed" in field_text
    assert "old_complete.json" not in field_text


def test_research_completion_notify_dry_run_does_not_post(tmp_path: Path, capsys) -> None:
    rc = research_completion_notify.main(
        [
            "--cadence",
            "daily",
            "--report-root",
            str(tmp_path / "reports"),
            "--channel",
            "#all-beatwallstreet",
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sent"] is False
    assert payload["reason"] == "dry_run"
    assert payload["payload"]["channel"] == "#all-beatwallstreet"


def test_research_completion_notify_posts_when_webhook_present(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, Mapping[str, Any], float]] = []

    def _fake_post(webhook_url: str, payload: Mapping[str, Any], timeout_s: float) -> int:
        calls.append((webhook_url, payload, timeout_s))
        return 200

    monkeypatch.setattr(research_completion_notify, "_post_slack_message", _fake_post)

    rc = research_completion_notify.main(
        [
            "--cadence",
            "weekly",
            "--report-root",
            str(tmp_path / "reports"),
            "--webhook-url",
            "https://hooks.slack.test/research",
            "--channel",
            "#all-beatwallstreet",
            "--timeout-s",
            "1.5",
        ]
    )

    assert rc == 0
    assert calls
    assert calls[0][0] == "https://hooks.slack.test/research"
    assert calls[0][1]["channel"] == "#all-beatwallstreet"
    assert calls[0][2] == 1.5


def test_research_completion_notify_hydrates_managed_webhook(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Mapping[str, Any], float]] = []

    def _fake_hydrate():
        monkeypatch.setattr(
            research_completion_notify,
            "_env_text",
            lambda name, default="": "https://hooks.slack.test/managed"
            if name == "AI_TRADING_SLACK_WEBHOOK_URL"
            else default,
        )
        return {"hydrated_count": 1}

    def _fake_post(webhook_url: str, payload: Mapping[str, Any], timeout_s: float) -> int:
        calls.append((webhook_url, payload, timeout_s))
        return 200

    monkeypatch.setattr(research_completion_notify, "hydrate_managed_secrets", _fake_hydrate)
    monkeypatch.setattr(research_completion_notify, "_post_slack_message", _fake_post)
    monkeypatch.setattr(research_completion_notify, "_env_text", lambda _name, default="": default)

    rc = research_completion_notify.main(
        [
            "--cadence",
            "daily",
            "--report-root",
            str(tmp_path / "reports"),
            "--channel",
            "#all-beatwallstreet",
        ]
    )

    assert rc == 0
    assert calls
    assert calls[0][0] == "https://hooks.slack.test/managed"
