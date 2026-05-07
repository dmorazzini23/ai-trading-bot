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
    )

    assert report["trade_allowed"] is False
    assert "runtime_gonogo_failed" in report["blocked_reasons"]


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
        memory_audit={"status": "ok"},
    )

    assert report["symbol_actions"]["shadow_promotion"]["available"] is True
    assert report["symbol_actions"]["shadow_promotion"]["suggestions"][0]["symbol"] == "AMZN"
    assert "AMZN" in daily_research_pipeline._markdown(report)


def test_daily_research_pipeline_cli_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    health = tmp_path / "health.json"
    live_cost = tmp_path / "live_cost.json"
    memory = tmp_path / "memory.json"
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
    assert payload["memory_status"]["status"] == "ok"
    assert latest.is_file()
    assert "Daily Research 2026-05-05" in md.read_text(encoding="utf-8")
