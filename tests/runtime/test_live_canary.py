from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import ai_trading.runtime.live_canary as live_canary
from ai_trading.runtime.live_canary import evaluate_launch_profile_order, observe_launch_profile_state
from ai_trading.telemetry import runtime_state


def _prime_runtime_state() -> None:
    runtime_state.update_data_provider_state(
        primary="alpaca-iex",
        active="alpaca-iex",
        using_backup=False,
        status="healthy",
        data_status="ready",
    )
    runtime_state.update_quote_status(
        allowed=True,
        symbol="AAPL",
        status="ready",
        source="alpaca",
        synthetic=False,
        bid=100.0,
        ask=100.05,
        quote_age_ms=100.0,
    )


def _approve_live_capital(monkeypatch, tmp_path: Path, *, status: str = "live_canary_allowed") -> None:
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_LIVE_CAPITAL_OPERATOR_APPROVED", "1")
    readiness_path = tmp_path / "runtime" / "live_capital_readiness_latest.json"
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(
        json.dumps({"artifact_type": "live_capital_readiness", "status": status}),
        encoding="utf-8",
    )


def _valid_live_order() -> dict:
    return {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1,
        "price_hint": 10.0,
        "quote_age_ms": 100.0,
        "spread_bps": 2.0,
        "daily_loss_state": {"daily_loss_abs": 0.0},
        "account_snapshot": {"id": "paper-account", "equity": 1000.0},
        "positions": [],
        "open_orders": [],
    }
































def test_live_restricted_enforces_launch_profile_caps_in_paper_rehearsal(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_restricted")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_RESTRICTED_MAX_NOTIONAL_PER_ORDER", "50")
    runtime_state.update_data_provider_state(
        primary="alpaca-iex",
        active="alpaca-iex",
        using_backup=False,
        status="healthy",
        data_status="ready",
    )
    runtime_state.update_quote_status(
        allowed=True,
        symbol="MSFT",
        status="ready",
        source="alpaca",
        synthetic=False,
        bid=100.0,
        ask=101.0,
        quote_age_ms=3000.0,
    )

    allowed, context = evaluate_launch_profile_order(
        {"symbol": "MSFT", "side": "sell_short", "quantity": 1, "price_hint": 100.0},
        execution_mode="paper",
    )

    assert allowed is False
    assert "symbol_not_allowlisted" in context["reasons"]
    assert "shorts_disabled" in context["reasons"]
    assert "notional_cap_exceeded" in context["reasons"]
    assert "quote_age_cap_exceeded" in context["reasons"]
    assert "spread_cap_exceeded" in context["reasons"]
    assert "operator_approval_missing" not in context["reasons"]
    state = observe_launch_profile_state()
    assert state["artifact_type"] == "launch_profile_state"
    assert state["profile"] == "live_restricted"


def test_live_profile_blocks_live_without_readiness_and_operator_approval(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_restricted")
    _prime_runtime_state()

    allowed, context = evaluate_launch_profile_order(
        {"symbol": "AAPL", "side": "buy", "quantity": 1, "price_hint": 10.0},
        execution_mode="live",
    )

    assert allowed is False
    assert "live_capital_readiness_not_allowed" in context["reasons"]
    assert "operator_approval_missing" in context["reasons"]


def test_launch_profile_gate_skips_non_runtime_execution_modes(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "paper_observe")

    allowed, context = evaluate_launch_profile_order(
        {"symbol": "AAPL", "side": "buy", "quantity": 1, "price_hint": 10.0},
        execution_mode="sim",
    )

    assert allowed is True
    assert context["enabled"] is False
    assert context["reason"] == "execution_mode_not_enforced"














