from __future__ import annotations

import json
from pathlib import Path

from ai_trading.runtime.live_canary import evaluate_canary_order, observe_live_canary_state
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


def test_live_canary_allows_tightly_bounded_allowlisted_order(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "1")
    _prime_runtime_state()

    allowed, context = evaluate_canary_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "price_hint": 50.0,
            "quote_age_ms": 100.0,
            "spread_bps": 5.0,
        },
        execution_mode="live",
    )

    assert allowed is True
    assert context["reasons"] == []
    state = observe_live_canary_state()
    assert state["entry_attempts"] == 1
    assert state["status"] == "ready"


def test_live_canary_blocks_symbol_short_notional_and_provider(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_NOTIONAL_PER_ORDER", "20")
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


def test_live_canary_blocks_after_daily_order_cap(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "1")
    _prime_runtime_state()

    order = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1,
        "price_hint": 10.0,
        "quote_age_ms": 100.0,
        "spread_bps": 2.0,
    }

    first_allowed, _ = evaluate_canary_order(order, execution_mode="live")
    second_allowed, second_context = evaluate_canary_order(order, execution_mode="live")

    assert first_allowed is True
    assert second_allowed is False
    assert "daily_order_count_cap_exceeded" in second_context["reasons"]


def test_live_canary_writes_state_and_event_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _prime_runtime_state()

    evaluate_canary_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "price_hint": 10.0,
            "quote_age_ms": 100.0,
            "spread_bps": 2.0,
        },
        execution_mode="live",
    )

    state_path = tmp_path / "runtime" / "live_canary_state_latest.json"
    events_path = tmp_path / "runtime" / "live_canary_events.jsonl"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[-1])
    assert state["artifact_type"] == "live_canary_state"
    assert event["allowed"] is True
