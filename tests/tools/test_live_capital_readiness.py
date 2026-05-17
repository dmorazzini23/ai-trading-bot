from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
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
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    report = live_capital_readiness.build_live_capital_readiness(
        health=_healthy_payload(),
        live_cost_model={"generated_at": now, "status": {"available": True, "status": "ready", "breach_count": 0}},
        promotion_report={"generated_at": now, "promotion_ready": True},
        validation={"generated_at": now, "full_validation_green": True},
        canary_plan={"generated_at": now, "paper_vs_live_canary_plan": "ready"},
        edge_calibration={"generated_at": now, "status": "calibrated"},
        execution_capture={"generated_at": now, "status": "acceptable"},
        portfolio_edge={"generated_at": now, "status": "ok"},
        pretrade_risk_verifier={"generated_at": now, "status": "passed"},
        post_trade_surveillance={"generated_at": now, "status": "clean"},
        walk_forward_capital={"generated_at": now, "status": "completed", "live_enabled": False},
        order_type_optimizer={"generated_at": now, "status": "ready", "live_enabled": False},
        regime_champions={"generated_at": now, "status": "ready"},
        adversarial_failure={"generated_at": now, "status": "passed", "live_money_authority": False},
        drift_monitor={"generated_at": now, "status": "ok"},
        approval_artifact={
            "generated_at": now,
            "approval_id": "approval-1",
            "status": "approved",
            "launch_profile": "live_canary",
        },
    )

    assert report["status"] == "live_canary_allowed"
    assert report["reasons"] == []
    assert report["gates"]["live_account_confirmed"] is True
    assert report["health_report_summary"]["status"] == "live_canary_allowed"
    assert report["canary_evidence"]["daily_research_mode"] is None
    assert report["canary_evidence"]["edge_calibration"]["status"] == "calibrated"
    assert report["canary_evidence"]["execution_capture"]["status"] == "acceptable"
    assert report["canary_evidence"]["portfolio_edge"]["status"] == "ok"
    assert report["canary_evidence"]["pretrade_risk_verifier"]["status"] == "passed"
    assert report["canary_evidence"]["model_data_drift_monitor"]["status"] == "ok"
    assert report["openclaw_summary"]["service"] == "ai-trading-live-capital"


def test_live_capital_readiness_requires_approval_artifact(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", "1")
    monkeypatch.setenv("AI_TRADING_LIVE_MAX_DAILY_LOSS", "25")
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    report = live_capital_readiness.build_live_capital_readiness(
        health=_healthy_payload(),
        live_cost_model={"generated_at": now, "status": {"available": True, "status": "ready", "breach_count": 0}},
        promotion_report={"generated_at": now, "promotion_ready": True},
        validation={"generated_at": now, "full_validation_green": True},
        canary_plan={"generated_at": now, "paper_vs_live_canary_plan": "ready"},
        edge_calibration={"generated_at": now, "status": "calibrated"},
        execution_capture={"generated_at": now, "status": "acceptable"},
        portfolio_edge={"generated_at": now, "status": "ok"},
        pretrade_risk_verifier={"generated_at": now, "status": "passed"},
        post_trade_surveillance={"generated_at": now, "status": "clean"},
        walk_forward_capital={"generated_at": now, "status": "completed", "live_enabled": False},
        order_type_optimizer={"generated_at": now, "status": "ready", "live_enabled": False},
        adversarial_failure={"generated_at": now, "status": "passed", "live_money_authority": False},
        drift_monitor={"generated_at": now, "status": "ok"},
    )

    assert report["status"] == "blocked"
    assert "live_capital_approval_artifact_missing" in report["reasons"]


def test_live_capital_readiness_rejects_stale_or_unscoped_approval(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", "1")
    monkeypatch.setenv("AI_TRADING_LIVE_MAX_DAILY_LOSS", "25")
    fresh = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    stale = (datetime.now(UTC) - timedelta(hours=30)).isoformat().replace("+00:00", "Z")

    report = live_capital_readiness.build_live_capital_readiness(
        health=_healthy_payload(),
        live_cost_model={"generated_at": fresh, "status": {"available": True, "status": "ready", "breach_count": 0}},
        promotion_report={"generated_at": fresh, "promotion_ready": True},
        validation={"generated_at": fresh, "full_validation_green": True},
        canary_plan={"generated_at": fresh, "paper_vs_live_canary_plan": "ready"},
        edge_calibration={"generated_at": fresh, "status": "calibrated"},
        execution_capture={"generated_at": fresh, "status": "acceptable"},
        portfolio_edge={"generated_at": fresh, "status": "ok"},
        pretrade_risk_verifier={"generated_at": fresh, "status": "passed"},
        post_trade_surveillance={"generated_at": fresh, "status": "clean"},
        walk_forward_capital={"generated_at": fresh, "status": "completed", "live_enabled": False},
        order_type_optimizer={"generated_at": fresh, "status": "ready", "live_enabled": False},
        adversarial_failure={"generated_at": fresh, "status": "passed", "live_money_authority": False},
        drift_monitor={"generated_at": fresh, "status": "ok"},
        approval_artifact={"generated_at": stale, "approval_id": "approval-1", "status": "approved"},
    )

    assert report["status"] == "blocked"
    assert "live_capital_approval_artifact_missing" in report["reasons"]
    assert report["approval"]["ok"] is False
    assert report["approval"]["fresh"]["fresh"] is False
    assert report["approval"]["scope_matches"] is False


def test_live_capital_readiness_blocks_stale_live_cost_evidence(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", "1")
    monkeypatch.setenv("AI_TRADING_LIVE_MAX_DAILY_LOSS", "25")
    fresh = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    stale = (datetime.now(UTC) - timedelta(hours=30)).isoformat().replace("+00:00", "Z")

    report = live_capital_readiness.build_live_capital_readiness(
        health=_healthy_payload(),
        live_cost_model={"generated_at": stale, "status": {"available": True, "breach_count": 0}},
        promotion_report={"generated_at": fresh, "promotion_ready": True},
        validation={"generated_at": fresh, "full_validation_green": True},
        canary_plan={"generated_at": fresh, "paper_vs_live_canary_plan": "ready"},
    )

    assert report["status"] == "blocked"
    assert "live_cost_model_stale" in report["reasons"]
    assert report["freshness"]["live_cost_model"]["fresh"] is False


def test_live_capital_readiness_blocks_provider_authority_and_daily_research(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", "1")
    monkeypatch.setenv("AI_TRADING_LIVE_MAX_DAILY_LOSS", "25")
    fresh = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    health = _healthy_payload()
    health["provider_authority"] = {"ok": False, "reasons": ["quote_source_unknown"]}

    report = live_capital_readiness.build_live_capital_readiness(
        health=health,
        live_cost_model={"generated_at": fresh, "status": {"available": True, "status": "ready", "breach_count": 0}},
        promotion_report={"generated_at": fresh, "promotion_ready": True},
        validation={"generated_at": fresh, "full_validation_green": True},
        canary_plan={
            "generated_at": fresh,
            "trade_allowed": False,
            "runtime_gonogo": {"gate_passed": False},
        },
    )

    assert report["status"] == "blocked"
    assert "provider_authority_not_ok" in report["reasons"]
    assert "daily_research_trade_not_allowed" in report["reasons"]
    assert "runtime_gonogo_failed" in report["reasons"]
    assert report["canary_evidence"]["daily_research_trade_allowed"] is False
    assert report["openclaw_summary"]["severity"] == "warning"


def test_live_capital_readiness_blocks_insufficient_cost_samples(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", "1")
    monkeypatch.setenv("AI_TRADING_LIVE_MAX_DAILY_LOSS", "25")
    fresh = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    report = live_capital_readiness.build_live_capital_readiness(
        health=_healthy_payload(),
        live_cost_model={"generated_at": fresh, "status": {"available": True, "status": "warming_up", "breach_count": 0}},
        promotion_report={"generated_at": fresh, "promotion_ready": True},
        validation={"generated_at": fresh, "full_validation_green": True},
        canary_plan={"generated_at": fresh, "trade_allowed": True},
    )

    assert report["status"] == "blocked"
    assert "live_cost_model_not_ready" in report["reasons"]


def test_live_capital_readiness_blocks_insufficient_samples_evidence(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LIVE_ACCOUNT_CONFIRMED", "1")
    monkeypatch.setenv("AI_TRADING_LIVE_MAX_DAILY_LOSS", "25")
    fresh = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    report = live_capital_readiness.build_live_capital_readiness(
        health=_healthy_payload(),
        live_cost_model={"generated_at": fresh, "status": {"available": True, "status": "ready", "breach_count": 0}},
        promotion_report={"generated_at": fresh, "promotion_ready": True},
        validation={"generated_at": fresh, "full_validation_green": True},
        canary_plan={"generated_at": fresh, "paper_vs_live_canary_plan": "ready"},
        edge_calibration={
            "generated_at": fresh,
            "status": "insufficient_samples",
            "sample_gate": {"sufficient": False},
        },
        execution_capture={"generated_at": fresh, "status": "acceptable"},
        portfolio_edge={"generated_at": fresh, "status": "ok"},
        pretrade_risk_verifier={"generated_at": fresh, "status": "passed"},
        post_trade_surveillance={"generated_at": fresh, "status": "clean"},
        walk_forward_capital={"generated_at": fresh, "status": "completed", "live_enabled": False},
        order_type_optimizer={"generated_at": fresh, "status": "ready", "live_enabled": False},
        adversarial_failure={"generated_at": fresh, "status": "passed", "live_money_authority": False},
        drift_monitor={"generated_at": fresh, "status": "ok"},
        approval_artifact={
            "generated_at": fresh,
            "approval_id": "approval-1",
            "status": "approved",
            "launch_profile": "live_canary",
        },
    )

    assert report["status"] == "blocked"
    assert "edge_calibration_insufficient_samples" in report["reasons"]


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
    assert payload["health_report_summary"]["status"] == "blocked"
