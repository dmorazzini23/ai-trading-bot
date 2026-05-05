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
    )

    assert report["trade_allowed"] is False
    assert "promotion_not_ready" in report["blocked_reasons"]
    assert report["recommended_next_session_mode"] == "paper_only"


def test_daily_research_pipeline_cli_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    health = tmp_path / "health.json"
    live_cost = tmp_path / "live_cost.json"
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

    rc = daily_research_pipeline.main(
        [
            "--report-date",
            "2026-05-05",
            "--health-json",
            str(health),
            "--live-cost-model-json",
            str(live_cost),
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
    assert latest.is_file()
    assert "Daily Research 2026-05-05" in md.read_text(encoding="utf-8")
