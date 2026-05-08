from __future__ import annotations

import json
from pathlib import Path

from ai_trading.runtime.live_canary import (
    evaluate_canary_order,
    evaluate_launch_profile_order,
    observe_launch_profile_state,
    observe_live_canary_state,
)
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
        source="latest_quote",
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


def test_live_canary_allows_tightly_bounded_allowlisted_order(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "1")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    allowed, context = evaluate_canary_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "price_hint": 50.0,
            "quote_age_ms": 100.0,
            "spread_bps": 5.0,
            "daily_loss_state": {"daily_loss_abs": 0.0},
        },
        execution_mode="live",
    )

    assert allowed is True
    assert context["reasons"] == []
    state = observe_live_canary_state()
    assert state["entry_attempts"] == 1
    assert state["status"] == "ready"


def test_live_canary_blocks_symbol_short_notional_and_provider(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_NOTIONAL_PER_ORDER", "20")
    _approve_live_capital(monkeypatch, tmp_path)
    runtime_state.update_data_provider_state(
        primary="alpaca-iex",
        active="yahoo",
        using_backup=True,
        status="degraded",
        data_status="degraded",
    )
    runtime_state.update_quote_status(
        allowed=False,
        symbol="MSFT",
        status="blocked",
        source="synthetic",
        synthetic=True,
    )

    allowed, context = evaluate_canary_order(
        {
            "symbol": "MSFT",
            "side": "sell_short",
            "quantity": 2,
            "price_hint": 30.0,
            "quote_age_ms": 5000.0,
            "spread_bps": 50.0,
        },
        execution_mode="live",
    )

    assert allowed is False
    assert "symbol_not_allowlisted" in context["reasons"]
    assert "shorts_disabled" in context["reasons"]
    assert "notional_cap_exceeded" in context["reasons"]
    assert "provider_degraded" in context["reasons"]
    assert "synthetic_quote" in context["reasons"]
    assert "quote_not_allowed" in context["reasons"]


def test_live_canary_blocks_after_daily_order_cap(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "1")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1,
        "price_hint": 10.0,
        "quote_age_ms": 100.0,
        "spread_bps": 2.0,
        "daily_loss_state": {"daily_loss_abs": 0.0},
    }

    first_allowed, _ = evaluate_canary_order(order, execution_mode="live")
    second_allowed, second_context = evaluate_canary_order(order, execution_mode="live")

    assert first_allowed is True
    assert second_allowed is False
    assert "daily_order_count_cap_exceeded" in second_context["reasons"]


def test_live_canary_blocks_missing_or_exceeded_daily_loss(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1,
        "price_hint": 10.0,
        "quote_age_ms": 100.0,
        "spread_bps": 2.0,
    }

    missing_allowed, missing_context = evaluate_canary_order(order, execution_mode="live")
    exceeded_allowed, exceeded_context = evaluate_canary_order(
        order | {"daily_loss_state": {"daily_loss_abs": 25.0}},
        execution_mode="live",
    )

    assert missing_allowed is False
    assert "daily_loss_state_missing" in missing_context["reasons"]
    assert exceeded_allowed is False
    assert "max_daily_loss_exceeded" in exceeded_context["reasons"]


def test_live_canary_derives_quote_age_and_spread_from_runtime_state(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_QUOTE_AGE_MS", "50")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_SPREAD_BPS", "5")
    _approve_live_capital(monkeypatch, tmp_path)
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
        source="latest_quote",
        synthetic=False,
        bid=100.0,
        ask=100.50,
        quote_age_ms=100.0,
    )

    allowed, context = evaluate_canary_order(
        {"symbol": "AAPL", "side": "buy", "quantity": 1, "price_hint": 10.0},
        execution_mode="live",
    )

    assert allowed is False
    assert "quote_age_cap_exceeded" in context["reasons"]
    assert "spread_cap_exceeded" in context["reasons"]


def test_live_canary_writes_state_and_event_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    evaluate_canary_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "price_hint": 10.0,
            "quote_age_ms": 100.0,
            "spread_bps": 2.0,
            "daily_loss_state": {"daily_loss_abs": 0.0},
        },
        execution_mode="live",
    )

    state_path = tmp_path / "runtime" / "live_canary_state_latest.json"
    events_path = tmp_path / "runtime" / "live_canary_events.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert state["artifact_type"] == "live_canary_state"
    assert event["allowed"] is True


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
        source="latest_quote",
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


def test_live_canary_blocks_projected_gross_and_symbol_exposure(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    allowed, context = evaluate_canary_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "price_hint": 60.0,
            "quote_age_ms": 100.0,
            "spread_bps": 2.0,
            "account_snapshot": {"equity": 1000.0},
            "positions": [{"symbol": "MSFT", "qty": 20, "market_price": 1.0}],
        },
        execution_mode="live",
    )

    assert allowed is False
    assert "max_gross_exposure_exceeded" in context["reasons"]
    assert "max_symbol_exposure_exceeded" in context["reasons"]
    assert context["exposure"]["evaluated"] is True


def test_live_canary_sell_to_close_is_not_short_and_reduces_exposure(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    allowed, context = evaluate_canary_order(
        {
            "symbol": "AAPL",
            "side": "sell",
            "quantity": 3,
            "price_hint": 10.0,
            "quote_age_ms": 100.0,
            "spread_bps": 2.0,
            "closing_position": True,
            "daily_loss_state": {"daily_loss_abs": 0.0},
            "account_snapshot": {"equity": 1000.0},
            "positions": [{"symbol": "AAPL", "qty": 4, "market_price": 10.0}],
        },
        execution_mode="live",
    )

    assert allowed is True
    assert "shorts_disabled" not in context["reasons"]
    assert "max_gross_exposure_exceeded" not in context["reasons"]
    assert "max_symbol_exposure_exceeded" not in context["reasons"]
