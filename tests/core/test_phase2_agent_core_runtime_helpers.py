from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from ai_trading.core import bot_engine
from ai_trading.core import execution_flow
from ai_trading.core import run_all_trades_execution as run_all_execution
from ai_trading.core.netting_symbol_approval import prepare_netting_symbol_approval
from ai_trading.core.netting_symbol_cycle import process_netting_symbol
from tests.test_netting_symbol_approval import _base_kwargs as _approval_base_kwargs
from tests.test_netting_symbol_cycle import (
    _make_net_target,
    _make_processor,
)


def _clear_replay_gate_cache() -> None:
    run_all_execution._REPLAY_LIVE_PARITY_GATE_CACHE["updated_mono"] = 0.0
    run_all_execution._REPLAY_LIVE_PARITY_GATE_CACHE["gate"] = None


def test_approval_cost_aware_entry_guard_blocks_weak_opening_edge() -> None:
    kwargs = _approval_base_kwargs()
    kwargs["current_shares"] = 0
    kwargs["delta_shares"] = 4
    kwargs["net_target"].proposals = [
        SimpleNamespace(expected_edge_bps=1.0, expected_cost_bps=4.0, confidence=0.8)
    ]
    kwargs["evaluate_execution_approval_func"] = lambda **_kwargs: (
        (_ for _ in ()).throw(AssertionError("approval should not be evaluated"))
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason == "COST_AWARE_ENTRY_GUARD"
    metrics = cast(dict[str, Any], result.blocked_metrics)["cost_aware_entry_guard"]
    assert metrics["expected_edge_bps_raw"] == 1.0
    assert metrics["expected_cost_bps"] == 4.0
    assert metrics["required_edge_bps"] > metrics["expected_edge_bps"]


def test_approval_min_notional_precheck_autosizes_payload_quantity() -> None:
    class _Engine:
        @staticmethod
        def _opening_min_notional_allows_order(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
            payload["quantity"] = 5
            payload["qty"] = 5
            return True, {"source": "autosize"}

    seen_delta: list[int] = []
    kwargs = _approval_base_kwargs()
    kwargs["current_shares"] = 0
    kwargs["delta_shares"] = 2
    kwargs["exec_engine"] = _Engine()
    kwargs["evaluate_execution_approval_func"] = lambda **approval_kwargs: (
        seen_delta.append(int(approval_kwargs["delta_shares"]))
        or SimpleNamespace(
            approval=SimpleNamespace(allowed=True, reasons=[], expected_net_edge_bps=12.0),
            adjusted_delta_shares=approval_kwargs["delta_shares"],
            adjusted_side=approval_kwargs["side"],
        )
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert seen_delta == [5]
    assert result.delta_shares == 5
    assert result.target_shares == 5
    assert "ENTRY_CONSTRAINED_MIN_NOTIONAL_AUTOSIZE_PRECHECK" in result.gates_added
    assert result.snapshot_updates["opening_min_notional_precheck"] == {
        "source": "autosize",
        "autosized_qty": 5,
    }


def test_approval_alpha_decay_block_can_be_bypassed_and_deweighted() -> None:
    kwargs = _approval_base_kwargs()
    kwargs["current_shares"] = 0
    kwargs["delta_shares"] = 10
    kwargs["alpha_decay_deweight_enabled"] = True
    kwargs["alpha_decay_qty_step"] = 0.2
    kwargs["alpha_decay_qty_max_deweight"] = 0.5
    kwargs["gate_blocks_func"] = lambda gate: gate != "ALPHA_DECAY_BLOCK"
    kwargs["alpha_decay_entry_guard_func"] = lambda *_args: {
        "blocked": True,
        "trades_in_window": 4,
        "start_trades": 2,
    }

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.delta_shares == 5
    assert result.gates_added[-2:] == (
        "ALPHA_DECAY_BLOCK_BYPASSED",
        "ALPHA_DECAY_DEWEIGHT",
    )
    assert result.snapshot_updates["alpha_decay"] == {
        "trades_in_window": 4,
        "start_trades": 2,
        "multiplier": 0.5,
    }


def test_approval_capacity_throttle_blocks_when_scaled_close_rounds_to_zero() -> None:
    kwargs = _approval_base_kwargs()
    kwargs["current_shares"] = 5
    kwargs["delta_shares"] = -1
    kwargs["capacity_throttle_enabled"] = True
    kwargs["capacity_spread_soft_bps"] = 1.0
    kwargs["capacity_spread_hard_bps"] = 2.0
    kwargs["capacity_min_scale"] = 0.1
    kwargs["liq_features"] = SimpleNamespace(spread_bps=5.0, rolling_volume=10_000.0)
    kwargs["clip_sell_qty_to_available_position_func"] = lambda **clip_kwargs: (
        int(clip_kwargs["requested_qty"]),
        {"available": 5},
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason == "CAPACITY_THROTTLE_BLOCK"
    assert result.gates_added == ("CAPACITY_THROTTLE_BLOCK",)
    metrics = cast(dict[str, Any], result.blocked_metrics)
    assert metrics["capacity_scale"] == pytest.approx(0.1)
    assert metrics["spread_bps"] == 5.0


def test_process_symbol_primary_feed_derisk_blocks_expanding_exposure() -> None:
    processor, records = _make_processor(
        primary_feed_derisk={"triggered": True, "block": True, "scale": 0.5},
        apply_symbol_adjustments_func=lambda **_kwargs: (
            (_ for _ in ()).throw(AssertionError("adjustments should not run"))
        ),
    )

    result = process_netting_symbol(
        processor=processor,
        symbol="AAPL",
        net_target=_make_net_target(bar_ts=processor.now, target_dollars=100.0),
        orders_submitted=0,
    )

    assert result.attempted_increment == 0
    assert records[0]["gates"] == ["DERISK_PRIMARY_FEED_BLOCK"]
    assert records[0]["config_snapshot"]["primary_feed_derisk"]["block"] is True


def test_process_symbol_event_blackout_failure_is_cached_false_and_continues() -> None:
    debug_events: list[str] = []
    cache: dict[str, bool] = {}
    processor, records = _make_processor(
        event_blackout_enabled=True,
        event_blackout_days=2,
        event_blackout_cache=cache,
        is_near_event_func=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("calendar down")),
        logger=SimpleNamespace(debug=lambda event, **_kwargs: debug_events.append(event)),
    )

    result = process_netting_symbol(
        processor=processor,
        symbol="AAPL",
        net_target=_make_net_target(bar_ts=processor.now),
        orders_submitted=0,
    )

    assert result.submitted_increment == 1
    assert cache == {"AAPL": False}
    assert debug_events == ["EVENT_BLACKOUT_CHECK_FAILED"]
    assert records[0]["config_snapshot"]["event_risk_near"] is False


def test_process_symbol_participation_block_is_not_bypassed_when_auto_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_TRADING_PARTICIPATION_CAP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PARTICIPATION_BLOCK_MODE", "block")
    processor, records = _make_processor(
        ineffective_gate_blocklist={"LIQ_PARTICIPATION_BLOCK", "LIQUIDITY_PARTICIPATION"},
        enforce_participation_cap_func=lambda **_kwargs: (
            False,
            2.0,
            "LIQ_PARTICIPATION_BLOCK",
        ),
    )

    result = process_netting_symbol(
        processor=processor,
        symbol="AAPL",
        net_target=_make_net_target(bar_ts=processor.now, target_dollars=100.0),
        orders_submitted=0,
    )

    assert result.attempted_increment == 0
    assert result.submitted_increment == 0
    assert records[0]["gates"] == ["LIQ_PARTICIPATION_BLOCK"]
    assert records[0]["net_target"].target_shares == 10
    assert "gate_auto_disable" not in records[0]["config_snapshot"]


def test_process_symbol_quarantine_zero_targets_flattens_without_blocking() -> None:
    class _QuarantineManager:
        @staticmethod
        def is_quarantined(**kwargs: Any) -> tuple[bool, str]:
            if kwargs.get("symbol") == "AAPL":
                return True, "SYMBOL_QUARANTINED"
            return False, ""

    processor, records = _make_processor(
        positions={"AAPL": 5},
        quarantine_enabled=True,
        quarantine_manager=_QuarantineManager(),
        quarantine_apply_symbol=True,
        quarantine_mode="zero_targets",
    )

    result = process_netting_symbol(
        processor=processor,
        symbol="AAPL",
        net_target=_make_net_target(bar_ts=processor.now, target_dollars=100.0),
        orders_submitted=0,
    )

    assert result.submitted_increment == 1
    assert "SYMBOL_QUARANTINED" in records[0]["gates"]
    assert records[0]["config_snapshot"]["quarantine_mode"] == "zero_targets"
    assert records[0]["net_target"].target_shares == 0


def test_send_exit_order_skips_symbol_absent_from_position_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "MarketOrderRequest", SimpleNamespace)
    monkeypatch.setattr(bot_engine, "LimitOrderRequest", SimpleNamespace)
    monkeypatch.setattr(bot_engine, "OrderSide", SimpleNamespace(SELL="sell"))
    monkeypatch.setattr(bot_engine, "TimeInForce", SimpleNamespace(DAY="day"))
    monkeypatch.setattr(
        bot_engine,
        "safe_submit_order",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not submit")),
    )
    ctx = SimpleNamespace(
        api=SimpleNamespace(
            get_position=lambda _symbol: (_ for _ in ()).throw(
                AssertionError("snapshot should short-circuit lookup")
            )
        )
    )

    execution_flow.send_exit_order(
        ctx,
        "AAPL",
        3,
        0.0,
        "manual_exit",
        raw_positions=[SimpleNamespace(symbol="MSFT", qty="3")],
    )


def test_pov_submit_continues_after_none_submission_and_fills_next_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.setattr(bot_engine, "fetch_minute_df_safe", lambda _symbol: pd.DataFrame({"volume": [10]}))
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "_ALPACA_IMPORT_ERROR", None, raising=False)
    monkeypatch.setattr(bot_engine, "StockLatestQuoteRequest", lambda **_kwargs: SimpleNamespace(), raising=False)

    submitted: list[int] = []

    def _submit_order(_ctx: Any, _symbol: str, qty: int, _side: str) -> Any:
        submitted.append(qty)
        if len(submitted) == 1:
            return None
        return SimpleNamespace(id="order-2", status="filled", filled_qty=str(qty))

    monkeypatch.setattr(bot_engine, "submit_order", _submit_order)
    ctx = SimpleNamespace(
        data_client=SimpleNamespace(
            get_stock_latest_quote=lambda _request: SimpleNamespace(ask_price=100.01, bid_price=100.0)
        )
    )
    cfg = SimpleNamespace(
        sleep_interval=1.0,
        max_retries=0,
        backoff_factor=2.0,
        max_backoff_interval=5.0,
        pct=0.10,
    )

    assert execution_flow.pov_submit(ctx, "AAPL", 1, "buy", cfg) is True

    assert submitted == [1, 1]
    assert not hasattr(ctx, "partial_fill_tracker")


def test_netting_cycle_always_ends_started_cycle_when_run_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_replay_gate_cache()
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "0")
    monkeypatch.setattr(bot_engine, "_decision_log_runtime_path", lambda: "runtime/test-decisions.jsonl")
    monkeypatch.setattr(bot_engine, "list_open_orders", lambda _api: [])
    monkeypatch.setattr(bot_engine, "_handle_pending_orders", lambda _orders, _runtime: False)
    monkeypatch.setattr(bot_engine, "_netting_pipeline_enabled", lambda _runtime: True)

    calls: list[str] = []

    class _Engine:
        @staticmethod
        def start_cycle() -> None:
            calls.append("start")

        @staticmethod
        def end_cycle() -> None:
            calls.append("end")

    def _run_netting_cycle(*_args: Any, **_kwargs: Any) -> None:
        calls.append("run")
        raise RuntimeError("cycle failed")

    monkeypatch.setattr(bot_engine, "_run_netting_cycle", _run_netting_cycle)
    runtime = SimpleNamespace(execution_engine=_Engine())

    with pytest.raises(RuntimeError, match="cycle failed"):
        run_all_execution.execute_run_all_trades_cycle(
            state=SimpleNamespace(),
            runtime=runtime,
            cfg_runtime=SimpleNamespace(post_submit_broker_sync=False),
            loop_id="loop-netting-raises",
            loop_start=0.0,
            api=SimpleNamespace(list_orders=lambda: []),
            restore_last_run_timestamp=lambda: None,
        )

    assert calls == ["start", "run", "end"]


def test_netting_cycle_preserves_down_provider_state_without_healthy_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_replay_gate_cache()
    monkeypatch.setenv("AI_TRADING_REPLAY_LIVE_PARITY_GATE_ENABLED", "0")
    monkeypatch.setenv("AI_TRADING_RUNTIME_GONOGO_REQUIRE_REPLAY_LIVE_PARITY_GATE", "0")
    monkeypatch.setattr(bot_engine, "_decision_log_runtime_path", lambda: "runtime/test-decisions.jsonl")
    monkeypatch.setattr(bot_engine, "list_open_orders", lambda _api: [])
    monkeypatch.setattr(bot_engine, "_handle_pending_orders", lambda _orders, _runtime: False)
    monkeypatch.setattr(bot_engine, "_netting_pipeline_enabled", lambda _runtime: True)
    monkeypatch.setattr(bot_engine, "_run_netting_cycle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_engine.runtime_state, "observe_data_provider_state", lambda: {"status": "down"})
    updates: list[dict[str, Any]] = []
    monkeypatch.setattr(
        bot_engine.runtime_state,
        "update_data_provider_state",
        lambda **kwargs: updates.append(kwargs),
    )

    run_all_execution.execute_run_all_trades_cycle(
        state=SimpleNamespace(),
        runtime=SimpleNamespace(),
        cfg_runtime=SimpleNamespace(post_submit_broker_sync=False),
        loop_id="loop-provider-down",
        loop_start=0.0,
        api=SimpleNamespace(list_orders=lambda: []),
        restore_last_run_timestamp=lambda: None,
    )

    assert updates == []
