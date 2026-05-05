from __future__ import annotations

import json
from pathlib import Path

from ai_trading.tools import live_capital_readiness


def _healthy_payload() -> dict[str, object]:
    return {
        "ok": True,
        "status": "ready",
        "broker": {"connected": True},
        "database": {"ok": True},
        "oms_invariants": {"ok": True},
        "oms_lifecycle_parity": {"ok": True},
        "replay_live_parity_gate": {"ok": True},
        "data_provider": {"status": "healthy"},
    }


def test_live_capital_readiness_blocks_without_live_cost(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", "1")

    report = live_capital_readiness.build_live_capital_readiness(
        health=_healthy_payload(),
        live_cost_model={},
        promotion_report={"promotion_ready": True},
        validation={"full_validation_green": True},
        canary_plan={"exists": True},
    )

    assert report["status"] == "blocked"
    assert "live_cost_model_unavailable" in report["reasons"]


def test_live_capital_readiness_allows_explicit_tiny_canary(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", "1")
    monkeypatch.setenv("AI_TRADING_LIVE_MAX_DAILY_LOSS", "25")

    report = live_capital_readiness.build_live_capital_readiness(
        health=_healthy_payload(),
        live_cost_model={"status": {"available": True, "breach_count": 0}},
        promotion_report={"promotion_ready": True},
        validation={"full_validation_green": True},
        canary_plan={"paper_vs_live_canary_plan": "ready"},
    )

    assert report["status"] == "live_canary_allowed"
    assert report["reasons"] == []
    assert report["gates"]["live_account_confirmed"] is True


def test_live_capital_readiness_cli_writes_blocked_artifact_with_success_override(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_trade")
    health = tmp_path / "health.json"
    out = tmp_path / "readiness.json"
    health.write_text(json.dumps(_healthy_payload()), encoding="utf-8")

    rc = live_capital_readiness.main(
        [
            "--health-json",
            str(health),
            "--output-json",
            str(out),
            "--success-on-blocked",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert "live_capital_profile_not_selected" in payload["reasons"]
