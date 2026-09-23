from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import ai_trading.runtime.live_canary as live_canary
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


def test_canary_corrupt_state_never_resets_exhausted_budget(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "1")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    assert evaluate_canary_order(_valid_live_order(), execution_mode="live")[0]
    state_path = tmp_path / "runtime" / "live_canary_state_latest.json"
    state_path.write_text("{invalid", encoding="utf-8")

    allowed, context = evaluate_canary_order(_valid_live_order(), execution_mode="live")

    assert not allowed
    assert context["reasons"] == ["launch_profile_state_corrupt"]
    assert state_path.read_text(encoding="utf-8") == "{invalid"
    assert observe_live_canary_state()["state_error"] == "launch_profile_state_corrupt"


def test_canary_invalid_count_and_future_state_block(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    assert evaluate_canary_order(_valid_live_order(), execution_mode="live")[0]
    state_path = tmp_path / "runtime" / "live_canary_state_latest.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_path.write_text(json.dumps(state | {"entry_attempts": "0"}), encoding="utf-8")
    assert evaluate_canary_order(_valid_live_order(), execution_mode="live")[1]["reasons"] == [
        "launch_profile_state_corrupt"
    ]
    future_day = (date.fromisoformat(state["date"]) + timedelta(days=1)).isoformat()
    state_path.write_text(json.dumps(state | {"date": future_day}), encoding="utf-8")
    assert evaluate_canary_order(_valid_live_order(), execution_mode="live")[1]["reasons"] == [
        "launch_profile_state_future_date"
    ]


def test_canary_missing_state_with_history_blocks(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    assert evaluate_canary_order(_valid_live_order(), execution_mode="live")[0]
    (tmp_path / "runtime" / "live_canary_state_latest.json").unlink()

    allowed, context = evaluate_canary_order(_valid_live_order(), execution_mode="live")

    assert not allowed
    assert context["reasons"] == ["launch_profile_state_missing_history"]


def test_canary_missing_snapshot_and_journal_after_initialization_blocks(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    assert evaluate_canary_order(_valid_live_order(), execution_mode="live")[0]
    runtime_dir = tmp_path / "runtime"
    assert (runtime_dir / "live_canary_state_latest.initialized").exists()
    (runtime_dir / "live_canary_state_latest.json").unlink()
    (runtime_dir / "live_canary_events.jsonl").unlink()

    allowed, context = evaluate_canary_order(_valid_live_order(), execution_mode="live")

    assert not allowed
    assert context["reasons"] == ["launch_profile_state_missing_history"]


def test_canary_valid_prior_day_state_rolls_over(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "1")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    assert evaluate_canary_order(_valid_live_order(), execution_mode="live")[0]
    state_path = tmp_path / "runtime" / "live_canary_state_latest.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    prior_day = (date.fromisoformat(state["date"]) - timedelta(days=1)).isoformat()
    state["date"] = prior_day
    state_path.write_text(json.dumps(state), encoding="utf-8")
    events_path = tmp_path / "runtime" / "live_canary_events.jsonl"
    event = json.loads(events_path.read_text(encoding="utf-8"))
    event["ts"] = prior_day + event["ts"][10:]
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    allowed, context = evaluate_canary_order(_valid_live_order(), execution_mode="live")

    assert allowed
    assert context["reasons"] == []
    assert observe_live_canary_state()["entry_attempts"] == 1


def test_canary_journal_preserves_attempt_after_snapshot_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "1")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    original_write = live_canary._write_state

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(live_canary, "_write_state", fail_write)
    allowed, context = evaluate_canary_order(_valid_live_order(), execution_mode="live")
    assert not allowed
    assert context["reasons"] == ["launch_profile_state_write_failed"]
    monkeypatch.setattr(live_canary, "_write_state", original_write)

    allowed, context = evaluate_canary_order(_valid_live_order(), execution_mode="live")
    assert not allowed
    assert context["reasons"] == ["launch_profile_state_missing_history"]


def test_canary_reconciles_journal_ahead_of_existing_state(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "2")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    assert evaluate_canary_order(_valid_live_order(), execution_mode="live")[0]
    original_write = live_canary._write_state

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(live_canary, "_write_state", fail_write)
    assert not evaluate_canary_order(_valid_live_order(), execution_mode="live")[0]
    monkeypatch.setattr(live_canary, "_write_state", original_write)

    allowed, context = evaluate_canary_order(_valid_live_order(), execution_mode="live")

    assert not allowed
    assert "daily_order_count_cap_exceeded" in context["reasons"]
    assert observe_live_canary_state()["entry_attempts"] == 2


def test_canary_concurrent_attempts_do_not_exceed_cap(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE_LIVE_CANARY_MAX_ORDER_COUNT", "1")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: evaluate_canary_order(_valid_live_order(), execution_mode="live"), range(4)))

    assert sum(allowed for allowed, _ in results) == 1
    assert observe_live_canary_state()["entry_attempts"] == 1


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
            "account_snapshot": {"id": "paper-account", "equity": 10000.0},
            "positions": [],
            "open_orders": [],
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
        "account_snapshot": {"id": "paper-account", "equity": 1000.0},
        "positions": [],
        "open_orders": [],
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
            "account_snapshot": {"id": "paper-account", "equity": 1000.0},
            "positions": [],
            "open_orders": [],
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


def test_live_canary_requires_complete_position_and_open_order_snapshots(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    allowed, context = evaluate_canary_order(
        {"symbol": "AAPL", "side": "buy", "quantity": 1, "price_hint": 10.0,
         "quote_age_ms": 100.0, "spread_bps": 2.0,
         "account_snapshot": {"equity": 1000.0},
         "daily_loss_state": {"daily_loss_abs": 0.0}},
        execution_mode="live",
    )
    assert allowed is False
    assert "account_identity_missing" in context["reasons"]
    assert "positions_snapshot_missing" in context["reasons"]
    assert "open_orders_snapshot_missing" in context["reasons"]


def test_live_canary_rejects_cost_basis_only_position_and_unpriced_pending_order(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    allowed, context = evaluate_canary_order(
        {"symbol": "AAPL", "side": "buy", "quantity": 1, "price_hint": 10.0,
         "quote_age_ms": 100.0, "spread_bps": 2.0,
         "account_snapshot": {"equity": 1000.0},
         "daily_loss_state": {"daily_loss_abs": 0.0},
         "positions": [{"symbol": "MSFT", "qty": 1, "avg_entry_price": 100.0}],
         "open_orders": [{"symbol": "MSFT", "remaining_qty": 1, "side": "buy"}]},
        execution_mode="live",
    )
    assert allowed is False
    assert "position_exposure_unverified" in context["reasons"]
    assert "open_order_exposure_unverified" in context["reasons"]


def test_live_canary_rejects_order_account_conflict(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()
    order = _valid_live_order() | {"account_id": "different-account"}

    allowed, context = evaluate_canary_order(order, execution_mode="live")
    assert allowed is False
    assert "account_identity_conflict" in context["reasons"]


def test_closing_flag_cannot_authorize_quantity_that_flips_long_to_short(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    _prime_runtime_state()

    allowed, context = evaluate_canary_order(
        {"symbol": "AAPL", "side": "sell", "quantity": 5, "price_hint": 10.0,
         "quote_age_ms": 100.0, "spread_bps": 2.0, "closing_position": True,
         "account_snapshot": {"equity": 1000.0},
         "daily_loss_state": {"daily_loss_abs": 0.0},
         "positions": [{"symbol": "AAPL", "qty": 4, "market_price": 10.0}],
         "open_orders": []},
        execution_mode="live",
    )
    assert allowed is False
    assert "shorts_disabled" in context["reasons"]


def test_live_canary_blocks_missing_quote_metadata_and_equity(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AI_TRADING_LAUNCH_PROFILE", "live_canary")
    _approve_live_capital(monkeypatch, tmp_path)
    runtime_state.reset_quote_status()
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
    )

    allowed, context = evaluate_canary_order(
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "price_hint": 10.0,
            "daily_loss_state": {"daily_loss_abs": 0.0},
        },
        execution_mode="live",
    )

    assert allowed is False
    assert "quote_age_unknown" in context["reasons"]
    assert "spread_unknown" in context["reasons"]
    assert "exposure_equity_missing" in context["reasons"]


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
