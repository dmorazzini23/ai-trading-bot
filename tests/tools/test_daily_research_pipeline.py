from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import daily_research_pipeline


def test_daily_research_report_blocks_live_profile_without_promotion(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={
            "ok": True,
            "status": "ready",
            "reason": "runtime_health_ok",
            "data_provider": {"status": "healthy"},
            "provider_authority": {"ok": True},
        },
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        shadow_report={"sample_gate": {"status": "ready"}},
        replay_governance={"replay_live_parity_gate": {"ok": True}},
        symbol_scorecard={"summary": {"allow": 2}, "symbols": []},
        promotion_report={"status": "blocked", "promotion_ready": False},
        runtime_gonogo={"gate_passed": True, "failed_checks": []},
        memory_audit={"status": "ok"},
        artifact_retention={"status": "ok"},
    )

    assert report["trade_allowed"] is False
    assert report["status"] == "blocked"
    assert "promotion_not_ready" in report["blocked_reasons"]
    assert report["recommended_next_session_mode"] == "paper_only"
    assert report["memory_status"]["status"] == "ok"


def test_daily_research_report_blocks_on_critical_memory(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={"ok": True, "status": "ready", "data_provider": {"status": "healthy"}},
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        replay_governance={"replay_live_parity_gate": {"ok": True}},
        runtime_gonogo={"gate_passed": True, "failed_checks": []},
        memory_audit={"status": "critical"},
    )

    assert report["trade_allowed"] is False
    assert "memory_status_critical" in report["blocked_reasons"]


def test_daily_research_report_blocks_on_runtime_gonogo_failure(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={"ok": True, "status": "ready", "data_provider": {"status": "healthy"}},
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        replay_governance={"replay_live_parity_gate": {"ok": True}},
        runtime_gonogo={"gate_passed": False, "failed_checks": ["win_rate"]},
        memory_audit={"status": "ok"},
        expected_edge_calibration={"status": "overestimated", "recommended_next_action": "keep_tiny_sampling"},
        evidence_starvation={"status": "starved", "recommendation": "widen_paper_diagnostic_sampling"},
        upward_trajectory={
            "status": "ready",
            "summary": {
                "recommended_next_action": "debug_validation_replay_gap_before_promotion",
                "candidate_count": 2,
            },
            "authority": {
                "runtime_authority": False,
                "promotion_authority": False,
                "live_money_authority": False,
            },
        },
        paper_sampling_state={"date": "2026-05-05", "count": 2},
    )

    assert report["trade_allowed"] is False
    assert report["status"] == "blocked"
    assert "runtime_gonogo_failed" in report["blocked_reasons"]
    assert report["expected_edge_calibration"]["status"] == "overestimated"
    assert report["evidence_starvation"]["recommendation"] == "widen_paper_diagnostic_sampling"
    assert report["diagnostic_sampling"]["paper_only"] is True
    assert report["diagnostic_sampling"]["state"]["count"] == 2
    assert report["health_report_summary"]["runtime_gonogo_passed"] is False
    assert report["next_level_artifacts"]["expected_edge_calibration"]["status"] == "overestimated"
    assert report["upward_trajectory"]["status"] == "ready"
    assert report["next_level_artifacts"]["upward_trajectory"]["candidate_count"] == 2
    assert report["next_level_artifacts"]["upward_trajectory"]["live_money_authority"] is False
    assert report["openclaw_summary"]["severity"] == "warning"


def test_daily_research_report_blocks_on_top_level_replay_governance_status(
    monkeypatch,
):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={"ok": True, "status": "ready", "data_provider": {"status": "healthy"}},
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        replay_governance={
            "status": "blocked",
            "reason": "REPLAY_POLICY_NON_REGRESSION_FAILED",
        },
        runtime_gonogo={"gate_passed": True, "failed_checks": []},
        memory_audit={"status": "ok"},
    )

    assert report["replay_status"]["status"] == "blocked"
    assert report["trade_allowed"] is False
    assert "replay_governance_failed" in report["blocked_reasons"]


def test_daily_research_report_blocks_on_provider_authority_failure(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={
            "ok": True,
            "status": "ready",
            "data_provider": {"status": "healthy"},
            "provider_authority": {"ok": False, "reasons": ["synthetic_quote"]},
        },
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        promotion_report={"status": "ready_for_approval", "promotion_ready": True},
        runtime_gonogo={"gate_passed": True, "failed_checks": []},
        memory_audit={"status": "ok"},
    )

    assert report["trade_allowed"] is False
    assert "provider_authority_not_ok" in report["blocked_reasons"]
    assert report["next_session_limits"]["profile"] == "live_canary"
    assert report["next_session_limits"]["provider_authority_ok"] is False


def test_daily_research_report_blocks_missing_live_cost(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={"ok": True, "status": "ready", "data_provider": {"status": "healthy"}},
        live_cost_model={},
        memory_audit={"status": "ok"},
    )

    assert report["trade_allowed"] is False
    assert "live_cost_unavailable" in report["blocked_reasons"]


def test_daily_research_report_reads_nested_runtime_gonogo_payload(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={"ok": True, "status": "ready", "data_provider": {"status": "healthy"}},
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        replay_governance={"replay_live_parity_gate": {"ok": True}},
        runtime_gonogo={
            "generated_at": "2026-05-05T20:00:00Z",
            "go_no_go": {
                "gate_passed": False,
                "failed_checks": ["win_rate", "live_samples_sufficient"],
            },
        },
        memory_audit={"status": "ok"},
    )

    assert report["trade_allowed"] is False
    assert report["runtime_gonogo"]["failed_checks"] == [
        "win_rate",
        "live_samples_sufficient",
    ]


def test_daily_research_report_blocks_on_market_closed_non_flat_position(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={
            "ok": True,
            "status": "healthy",
            "attention_flags": ["market_closed_non_flat_positions"],
            "data_provider": {"status": "warming_up"},
        },
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        replay_governance={"replay_live_parity_gate": {"ok": True}},
        runtime_gonogo={"gate_passed": True, "failed_checks": []},
        memory_audit={"status": "ok"},
    )

    assert report["trade_allowed"] is False
    assert "market_closed_non_flat_positions" in report["blocked_reasons"]
    assert report["runtime_attention_flags"] == ["market_closed_non_flat_positions"]


def test_daily_research_report_surfaces_shadow_promotion(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={"ok": True, "status": "ready", "data_provider": {"status": "healthy"}},
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        symbol_scorecard={
            "summary": {"allow": 2},
            "policy": {"allowed_symbols": ["AAPL"]},
            "shadow_promotion": {
                "available": True,
                "suggestions": [{"symbol": "AMZN"}],
            },
            "symbols": [],
        },
        symbol_promotion_comparison={
            "status": "ready",
            "promotion_authority": False,
            "runtime_symbol_gating_changed": False,
            "summary": {"symbol_count": 1},
            "symbols": [
                {
                    "symbol": "AMZN",
                    "recommendation": "consider_promotion",
                    "confidence": "high",
                }
            ],
        },
        memory_audit={"status": "ok"},
    )

    assert report["symbol_actions"]["shadow_promotion"]["available"] is True
    assert report["symbol_actions"]["shadow_promotion"]["suggestions"][0]["symbol"] == "AMZN"
    assert report["symbol_promotion"]["promotion_authority"] is False
    assert report["symbol_promotion"]["runtime_symbol_gating_changed"] is False
    assert report["symbol_promotion"]["digest"] == "AMZN:consider_promotion/high"
    assert "AMZN" in daily_research_pipeline._markdown(report)


def test_daily_research_report_surfaces_next_level_artifacts(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")

    report = daily_research_pipeline.build_daily_research_report(
        report_date="2026-05-05",
        health={"ok": True, "status": "ready", "data_provider": {"status": "healthy"}},
        live_cost_model={"status": {"available": True, "breach_count": 0, "status": "ready"}},
        replay_governance={"replay_live_parity_gate": {"ok": True}},
        symbol_lifecycle={
            "status": "ready",
            "summary": {"recommendations": {"consider_canary": 1}},
            "symbols": [{"symbol": "AMZN", "recommendation": "consider_canary"}],
        },
        execution_capture={"status": "acceptable", "summary": {"classification_counts": {}}},
        counterfactual_execution={"status": "passed", "summary": {"missed_positive_count": 0}},
        portfolio_edge={"status": "ok", "summary": {"portfolio_capture_ratio": 0.7}},
        decision_receipts={"status": "complete", "summary": {"receipts": 3}},
        huggingface_discovery={
            "status": "discovered",
            "summary": {"candidate_count": 4, "accepted_for_offline_experiment": 1},
        },
        huggingface_candidate_intake={
            "status": "ready_for_manual_review",
            "summary": {"accepted_for_offline_experiment": 1, "blocked": 2},
        },
        weekend_research={
            "status": "complete",
            "cadence": "weekend-sunday",
            "monday_preparation": {"recommended_operator_action": "review_monday_preparation_before_market_open"},
        },
        memory_audit={"status": "ok"},
    )

    assert report["symbol_lifecycle"]["status"] == "ready"
    assert report["execution_capture"]["status"] == "acceptable"
    assert report["counterfactual_execution"]["status"] == "passed"
    assert report["portfolio_edge_control"]["output"] == "ok"
    assert report["decision_receipts"]["summary"]["receipts"] == 3
    assert report["huggingface_research"]["status"]["discovery"] == "discovered"
    assert report["huggingface_research"]["summary"]["accepted_for_offline_experiment"] == 1
    assert report["huggingface_research"]["runtime_authority"] is False
    assert report["next_level_artifacts"]["huggingface_research"]["runtime_authority"] is False
    assert report["weekend_research"]["research_only"] is True
    assert report["next_level_artifacts"]["weekend_research"]["live_money_authority"] is False


def test_daily_research_pipeline_cli_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    health = tmp_path / "health.json"
    live_cost = tmp_path / "live_cost.json"
    memory = tmp_path / "memory.json"
    pretrade = tmp_path / "pretrade.json"
    symbol_promotion = tmp_path / "symbol_promotion.json"
    calibration = tmp_path / "calibration.json"
    starvation = tmp_path / "starvation.json"
    paper_sampling = tmp_path / "paper_sampling.json"
    hf_discovery = tmp_path / "hf_discovery.json"
    out = tmp_path / "daily.json"
    latest = tmp_path / "latest.json"
    md = tmp_path / "daily.md"
    health.write_text(
        json.dumps({"ok": True, "status": "ready", "data_provider": {"status": "healthy"}}),
        encoding="utf-8",
    )
    live_cost.write_text(
        json.dumps({"status": {"available": True, "breach_count": 0, "status": "ready"}}),
        encoding="utf-8",
    )
    memory.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    pretrade.write_text(
        json.dumps({"status": "passed", "fail_closed": False, "summary": {"violations": 0}}),
        encoding="utf-8",
    )
    symbol_promotion.write_text(
        json.dumps(
            {
                "status": "ready",
                "promotion_authority": False,
                "symbols": [
                    {
                        "symbol": "MSFT",
                        "recommendation": "collect_more_evidence",
                        "confidence": "low",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calibration.write_text(json.dumps({"status": "calibrated"}), encoding="utf-8")
    starvation.write_text(
        json.dumps({"status": "collecting", "recommendation": "keep_sampling"}),
        encoding="utf-8",
    )
    paper_sampling.write_text(json.dumps({"date": "2026-05-05", "count": 1}), encoding="utf-8")
    hf_discovery.write_text(
        json.dumps({"status": "discovered", "summary": {"candidate_count": 2}}),
        encoding="utf-8",
    )

    rc = daily_research_pipeline.main(
        [
            "--report-date",
            "2026-05-05",
            "--health-json",
            str(health),
            "--live-cost-model-json",
            str(live_cost),
                "--memory-audit-json",
                str(memory),
                "--pretrade-risk-json",
                str(pretrade),
                "--symbol-promotion-json",
                str(symbol_promotion),
            "--expected-edge-calibration-json",
            str(calibration),
            "--evidence-starvation-json",
            str(starvation),
            "--paper-sampling-state-json",
            str(paper_sampling),
            "--huggingface-discovery-json",
            str(hf_discovery),
            "--output-json",
            str(out),
            "--latest-json",
            str(latest),
            "--output-md",
            str(md),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "daily_research_report"
    assert payload["trade_allowed"] is True
    assert payload["status"] == "ready"
    assert payload["memory_status"]["status"] == "ok"
    assert payload["symbol_promotion"]["digest"] == "MSFT:collect_more_evidence/low"
    assert payload["expected_edge_calibration"]["status"] == "calibrated"
    assert payload["evidence_starvation"]["status"] == "collecting"
    assert payload["diagnostic_sampling"]["state"]["count"] == 1
    assert payload["huggingface_research"]["summary"]["candidates_scanned"] == 2
    assert payload["health_report_summary"]["trade_allowed"] is True
    assert payload["next_level_artifacts"]["diagnostic_sampling"]["paper_only"] is True
    assert payload["openclaw_summary"]["service"] == "ai-trading-research"
    assert latest.is_file()
    markdown = md.read_text(encoding="utf-8")
    assert "Daily Research 2026-05-05" in markdown
    assert "Expected-edge calibration" in markdown
    assert "Hugging Face research" in markdown
