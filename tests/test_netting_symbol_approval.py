from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai_trading.core.execution_guards import ExecutionApprovalContext
from ai_trading.core.netting_symbol_approval import prepare_netting_symbol_approval
from ai_trading.policy.compiler import ExecutionApproval


def _base_kwargs() -> dict[str, Any]:
    return {
        "state": SimpleNamespace(operational_safety_tier="normal"),
        "symbol": "AAPL",
        "now": datetime(2026, 4, 19, 15, 30, tzinfo=UTC),
        "current_shares": 5,
        "delta_shares": -10,
        "price": 100.0,
        "net_target": SimpleNamespace(
            proposals=[
                SimpleNamespace(expected_edge_bps=12.0, expected_cost_bps=4.0, confidence=0.8),
            ]
        ),
        "liq_features": SimpleNamespace(spread_bps=5.0, rolling_volume=10000.0),
        "liq_regime": SimpleNamespace(name="NORMAL"),
        "exec_engine": None,
        "effective_policy": SimpleNamespace(
            objective=SimpleNamespace(fee_bps=0.5, borrow_bps=0.0),
            calibration=SimpleNamespace(
                min_samples=10,
                max_ece_stress=0.3,
                max_brier_stress=0.4,
                max_ece_normal=0.2,
                max_brier_normal=0.3,
            ),
        ),
        "candidate_expected_net_edge": {"AAPL": 15.0},
        "edge_realism_rank_factor_by_symbol": {"AAPL": 1.0},
        "edge_realism_apply_to_approval_enabled": False,
        "alpha_decay_deweight_enabled": False,
        "alpha_decay_qty_step": 0.1,
        "alpha_decay_qty_max_deweight": 0.5,
        "capacity_throttle_enabled": False,
        "capacity_spread_soft_bps": 8.0,
        "capacity_spread_hard_bps": 20.0,
        "capacity_volume_soft_participation": 0.1,
        "capacity_volume_hard_participation": 0.2,
        "capacity_min_scale": 0.25,
        "slo_derisk_scale": 1.0,
        "slo_derisk_details": {},
        "primary_feed_derisk": {},
        "feed_derisk_scale": 1.0,
        "portfolio_current_gross": 10000.0,
        "sector_gross": {"TECH": 5000.0},
        "max_new_orders_per_cycle": 5,
        "orders_submitted": 0,
        "gates": [],
        "symbol_snapshot": {},
        "gate_blocks_func": lambda gate: True,
        "clip_sell_qty_to_available_position_func": lambda **kwargs: (0, {"available": 0}),
        "percentile_linear_func": lambda values, pct: 9.0,
        "slippage_setting_bps_func": lambda: 1.0,
        "safe_float_func": lambda value: float(value) if value is not None else None,
        "get_sector_func": lambda symbol: "TECH",
        "alpha_decay_entry_guard_func": lambda state, symbol, now: {},
        "evaluate_execution_approval_func": lambda **kwargs: SimpleNamespace(
            approval=SimpleNamespace(allowed=True, reasons=[], expected_net_edge_bps=11.0),
            adjusted_delta_shares=kwargs["delta_shares"],
            adjusted_side=kwargs["side"],
        ),
        "approve_execution_candidate_func": lambda *args, **kwargs: None,
    }


def test_prepare_netting_symbol_approval_blocks_when_sell_qty_unavailable() -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 5
    kwargs["delta_shares"] = -3

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason == "PRE_SUBMIT_INSUFFICIENT_POSITION_AVAILABLE"
    assert result.gates_added == ("PRE_SUBMIT_INSUFFICIENT_POSITION_AVAILABLE",)
    assert result.blocked_metrics == {"pre_submit_sell_qty_clip": {"available": 0}}


def test_prepare_netting_symbol_approval_applies_approval_adjustment() -> None:
    kwargs = _base_kwargs()
    kwargs["delta_shares"] = 8
    kwargs["current_shares"] = 0
    kwargs["exec_engine"] = SimpleNamespace()
    kwargs["clip_sell_qty_to_available_position_func"] = lambda **kwargs: (
        kwargs["requested_qty"],
        {},
    )
    kwargs["evaluate_execution_approval_func"] = lambda **kwargs: SimpleNamespace(
        approval=SimpleNamespace(allowed=True, reasons=["APPROVAL_SCALE"], expected_net_edge_bps=7.5),
        adjusted_delta_shares=3,
        adjusted_side="buy",
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.delta_shares == 3
    assert result.target_shares == 3
    assert result.target_dollars == 300.0
    assert result.side == "buy"
    assert result.opening_trade is True
    assert result.gates_added == ("APPROVAL_SCALE",)


def test_prepare_netting_symbol_approval_restores_one_share_paper_soft_throttle(
    monkeypatch,
) -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 0
    kwargs["delta_shares"] = 1
    kwargs["exec_engine"] = SimpleNamespace()
    kwargs["clip_sell_qty_to_available_position_func"] = lambda **kwargs: (
        kwargs["requested_qty"],
        {},
    )
    kwargs["evaluate_execution_approval_func"] = lambda **kwargs: ExecutionApprovalContext(
        approval=ExecutionApproval(
            False,
            0,
            1.5,
            ("RISK_FACTOR_SOFT_THROTTLE", "ZERO_QTY"),
        ),
        adjusted_delta_shares=0,
        adjusted_side="buy",
        pacing_headroom=5,
        stale_orders_present=False,
        portfolio_post_gross=100.0,
        factor_post_ratio=1.0,
        sector_name="TECH",
    )
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_PAPER_SAMPLING_SOFT_THROTTLE_FLOOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PAPER_SAMPLING_MIN_EXPECTED_NET_EDGE_BPS", "0.10")

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.delta_shares == 1
    assert result.target_shares == 1
    assert result.side == "buy"
    assert "PAPER_SAMPLE_SOFT_THROTTLE_FLOOR" in result.gates_added
    assert result.approval is not None
    assert result.approval.allowed is True


def test_prepare_netting_symbol_approval_keeps_live_soft_throttle_blocked(
    monkeypatch,
) -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 0
    kwargs["delta_shares"] = 1
    kwargs["exec_engine"] = SimpleNamespace()
    kwargs["clip_sell_qty_to_available_position_func"] = lambda **kwargs: (
        kwargs["requested_qty"],
        {},
    )
    kwargs["evaluate_execution_approval_func"] = lambda **kwargs: ExecutionApprovalContext(
        approval=ExecutionApproval(
            False,
            0,
            1.5,
            ("RISK_FACTOR_SOFT_THROTTLE", "ZERO_QTY"),
        ),
        adjusted_delta_shares=0,
        adjusted_side="buy",
        pacing_headroom=5,
        stale_orders_present=False,
        portfolio_post_gross=100.0,
        factor_post_ratio=1.0,
        sector_name="TECH",
    )
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_PAPER_SAMPLING_SOFT_THROTTLE_FLOOR_ENABLED", "1")

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason == "EXECUTION_APPROVAL_BLOCK"
    assert "PAPER_SAMPLE_SOFT_THROTTLE_FLOOR" not in result.gates_added


def test_prepare_netting_symbol_approval_allows_opening_short_without_long_inventory() -> None:
    clip_calls: list[dict[str, Any]] = []
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 0
    kwargs["delta_shares"] = -4
    kwargs["exec_engine"] = SimpleNamespace()

    def _clip_sell_qty_to_available_position(**clip_kwargs: Any) -> tuple[int, dict[str, int]]:
        clip_calls.append(dict(clip_kwargs))
        return 0, {"available": 0}

    kwargs["clip_sell_qty_to_available_position_func"] = _clip_sell_qty_to_available_position

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.side == "sell_short"
    assert result.delta_shares == -4
    assert result.target_shares == -4
    assert clip_calls == []


def test_prepare_netting_symbol_approval_preserves_sell_short_after_approval_adjustment() -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 0
    kwargs["delta_shares"] = -10
    kwargs["exec_engine"] = SimpleNamespace()
    kwargs["clip_sell_qty_to_available_position_func"] = lambda **_kwargs: (
        0,
        {"available": 0},
    )
    kwargs["evaluate_execution_approval_func"] = lambda **_kwargs: SimpleNamespace(
        approval=SimpleNamespace(allowed=True, reasons=["APPROVAL_SCALE"], expected_net_edge_bps=7.5),
        adjusted_delta_shares=-3,
        adjusted_side="sell",
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.side == "sell_short"
    assert result.delta_shares == -3
    assert result.target_shares == -3


def test_prepare_netting_symbol_approval_closes_long_before_opening_short() -> None:
    clip_calls: list[dict[str, Any]] = []
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 5
    kwargs["delta_shares"] = -10
    kwargs["exec_engine"] = SimpleNamespace()

    def _clip_sell_qty_to_available_position(**clip_kwargs: Any) -> tuple[int, dict[str, int]]:
        clip_calls.append(dict(clip_kwargs))
        return min(int(clip_kwargs["requested_qty"]), 5), {"available": 5}

    kwargs["clip_sell_qty_to_available_position_func"] = _clip_sell_qty_to_available_position

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.side == "sell"
    assert result.delta_shares == -5
    assert result.target_shares == 0
    assert result.opening_trade is False
    assert result.gates_added == ("PRE_SUBMIT_SELL_QTY_CLIP_AVAILABLE_POSITION",)
    assert clip_calls[0]["requested_qty"] == 10


def test_prepare_netting_symbol_approval_covers_short_before_opening_long() -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = -5
    kwargs["delta_shares"] = 10
    kwargs["exec_engine"] = SimpleNamespace()

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.side == "buy"
    assert result.delta_shares == 5
    assert result.target_shares == 0
    assert result.opening_trade is False
    assert result.gates_added == ("PRE_SUBMIT_BUY_QTY_CLIP_SHORT_COVER",)
    assert result.snapshot_updates["pre_submit_buy_qty_clip"] == {
        "current_shares": -5,
        "requested_qty": 10,
        "max_cover_qty": 5,
    }


def test_prepare_netting_symbol_approval_clips_post_approval_long_to_short_adjustment() -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 5
    kwargs["delta_shares"] = -3
    kwargs["exec_engine"] = SimpleNamespace()
    kwargs["clip_sell_qty_to_available_position_func"] = lambda **clip_kwargs: (
        min(int(clip_kwargs["requested_qty"]), 5),
        {"available": 5},
    )
    kwargs["evaluate_execution_approval_func"] = lambda **_kwargs: SimpleNamespace(
        approval=SimpleNamespace(allowed=True, reasons=["APPROVAL_SCALE"], expected_net_edge_bps=7.5),
        adjusted_delta_shares=-8,
        adjusted_side="sell_short",
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.side == "sell"
    assert result.delta_shares == -5
    assert result.target_shares == 0
    assert result.opening_trade is False
    assert result.gates_added == (
        "APPROVAL_SCALE",
        "PRE_SUBMIT_SELL_QTY_CLIP_AVAILABLE_POSITION",
    )


def test_prepare_netting_symbol_approval_clips_post_approval_short_to_long_adjustment() -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = -5
    kwargs["delta_shares"] = 3
    kwargs["exec_engine"] = SimpleNamespace()
    kwargs["evaluate_execution_approval_func"] = lambda **_kwargs: SimpleNamespace(
        approval=SimpleNamespace(allowed=True, reasons=["APPROVAL_SCALE"], expected_net_edge_bps=7.5),
        adjusted_delta_shares=8,
        adjusted_side="buy",
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.side == "buy"
    assert result.delta_shares == 5
    assert result.target_shares == 0
    assert result.opening_trade is False
    assert result.gates_added == (
        "APPROVAL_SCALE",
        "PRE_SUBMIT_BUY_QTY_CLIP_SHORT_COVER",
    )


def test_prepare_netting_symbol_approval_keeps_flip_close_as_sell_after_throttle() -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 5
    kwargs["delta_shares"] = -10
    kwargs["exec_engine"] = SimpleNamespace()
    kwargs["capacity_throttle_enabled"] = True
    kwargs["capacity_spread_soft_bps"] = 5.0
    kwargs["capacity_spread_hard_bps"] = 10.0
    kwargs["capacity_min_scale"] = 0.5
    kwargs["liq_features"] = SimpleNamespace(spread_bps=10.0, rolling_volume=10000.0)
    kwargs["clip_sell_qty_to_available_position_func"] = lambda **clip_kwargs: (
        min(int(clip_kwargs["requested_qty"]), 5),
        {"available": 5},
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.side == "sell"
    assert result.delta_shares == -2
    assert result.target_shares == 3
    assert result.opening_trade is False
    assert "PRE_SUBMIT_SELL_QTY_CLIP_AVAILABLE_POSITION" in result.gates_added
    assert "CAPACITY_THROTTLE_SCALE" in result.gates_added
    assert result.snapshot_updates["capacity_throttle"]["before_delta_shares"] == -5
    assert result.snapshot_updates["capacity_throttle"]["after_delta_shares"] == -2


def test_prepare_netting_symbol_approval_still_throttles_an_add() -> None:
    kwargs = _base_kwargs()
    kwargs["current_shares"] = 5
    kwargs["delta_shares"] = 5
    kwargs["exec_engine"] = SimpleNamespace()
    kwargs["capacity_throttle_enabled"] = True
    kwargs["capacity_spread_soft_bps"] = 1.0
    kwargs["capacity_spread_hard_bps"] = 2.0
    kwargs["capacity_min_scale"] = 0.5
    kwargs["liq_features"] = SimpleNamespace(spread_bps=5.0, rolling_volume=10000.0)

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.side == "buy"
    assert result.delta_shares == 2
    assert "CAPACITY_THROTTLE_SCALE" in result.gates_added
    assert "CLOSING_EXIT_OVERRIDE" not in result.gates_added


# ---- exits under the compiler's soft throttles (real approve_execution_candidate) ----


def _risk_policy(**budgets: float) -> Any:
    big = 1e12
    risk_budgets = {
        "symbol_hard_dollars": big,
        "symbol_soft_dollars": big,
        "portfolio_hard_gross_dollars": big,
        "portfolio_soft_gross_dollars": big,
        "sleeve_hard_dollars": big,
        "sleeve_soft_dollars": big,
        "factor_soft_ratio": 10.0,
        "factor_hard_ratio": 20.0,
    }
    risk_budgets.update(budgets)
    return SimpleNamespace(
        objective=SimpleNamespace(fee_bps=0.0, borrow_bps=0.0, min_expected_net_edge_bps=-1000.0),
        execution=SimpleNamespace(
            stale_order_block_sec=1e9, max_spread_bps=1000.0, min_rolling_volume=0.0, calibration_block_enabled=False
        ),
        risk_budgets=SimpleNamespace(**risk_budgets),
        calibration=SimpleNamespace(
            max_ece_normal=1.0, max_brier_normal=1.0, max_ece_stress=1.0, max_brier_stress=1.0, min_samples=10
        ),
        safety=SimpleNamespace(attack_size_multiplier=1.0),
    )


def _real_approval_kwargs(current: int, delta: int, policy: Any, **extra: Any) -> dict[str, Any]:
    from ai_trading.core.execution_guards import evaluate_execution_approval
    from ai_trading.policy.compiler import approve_execution_candidate

    kwargs = _base_kwargs()
    kwargs.update(
        current_shares=current,
        delta_shares=delta,
        price=10.0,
        effective_policy=policy,
        evaluate_execution_approval_func=evaluate_execution_approval,
        approve_execution_candidate_func=approve_execution_candidate,
        clip_sell_qty_to_available_position_func=lambda **k: (int(k["requested_qty"]), None),
        gate_blocks_func=lambda gate: gate != "COST_AWARE_ENTRY_GUARD",
    )
    kwargs.update(extra)
    return kwargs


@pytest.mark.parametrize("protective", [False, True], ids=["closing", "protective"])
def test_soft_throttle_rounding_an_exit_to_zero_keeps_it_whole(protective: bool) -> None:
    # Close 1 share with portfolio gross far above its soft limit: the throttle
    # scales the order to 0 (ZERO_QTY), which used to block the exit outright.
    kwargs = _real_approval_kwargs(
        1,
        -1,
        _risk_policy(portfolio_soft_gross_dollars=1000.0),
        portfolio_current_gross=100_000.0,
        protective_exit=protective,
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    marker = "PROTECTIVE_EXIT_OVERRIDE" if protective else "CLOSING_EXIT_OVERRIDE"
    assert result.blocked_reason is None
    assert result.delta_shares == -1
    assert "RISK_PORTFOLIO_SOFT_THROTTLE" in result.gates_added
    assert marker in result.gates_added


@pytest.mark.parametrize("protective", [False, True], ids=["trim", "protective"])
def test_soft_throttle_rounding_a_one_share_trim_to_zero(protective: bool) -> None:
    # Long 100 at $10, sell 1: post notional $990 against a $100 soft limit
    # rounds the order to 0. A partial trim stays blocked as on main; a
    # protective exit of the same size goes out.
    kwargs = _real_approval_kwargs(100, -1, _risk_policy(symbol_soft_dollars=100.0), protective_exit=protective)

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert "RISK_SYMBOL_SOFT_THROTTLE" in result.gates_added
    if protective:
        assert result.blocked_reason is None
        assert result.delta_shares == -1
        assert "PROTECTIVE_EXIT_OVERRIDE" in result.gates_added
    else:
        assert result.blocked_reason == "EXECUTION_APPROVAL_BLOCK"
        assert "ZERO_QTY" in result.gates_added
        assert not {"CLOSING_EXIT_OVERRIDE", "PROTECTIVE_EXIT_OVERRIDE"} & set(result.gates_added)


def test_soft_throttle_shrinking_a_full_close_is_restored_and_a_trim_is_scaled() -> None:
    close = prepare_netting_symbol_approval(
        **cast(
            Any,
            _real_approval_kwargs(
                100, -100, _risk_policy(portfolio_soft_gross_dollars=1000.0), portfolio_current_gross=4000.0
            ),
        )
    )
    trim = prepare_netting_symbol_approval(
        **cast(Any, _real_approval_kwargs(100, -50, _risk_policy(symbol_soft_dollars=100.0)))
    )

    assert close.blocked_reason is None
    assert close.delta_shares == -100
    assert "RISK_PORTFOLIO_SOFT_THROTTLE" in close.gates_added
    assert "CLOSING_EXIT_OVERRIDE" in close.gates_added
    # The trim is scaled exactly as on main: $500 post notional over a $100 limit.
    assert trim.blocked_reason is None
    assert trim.delta_shares == -10
    assert "RISK_SYMBOL_SOFT_THROTTLE" in trim.gates_added
    assert "CLOSING_EXIT_OVERRIDE" not in trim.gates_added


def test_hard_block_with_a_soft_throttle_still_holds_a_protective_exit() -> None:
    kwargs = _real_approval_kwargs(
        100,
        -10,
        _risk_policy(symbol_soft_dollars=100.0, portfolio_hard_gross_dollars=1.0),
        protective_exit=True,
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason == "EXECUTION_APPROVAL_BLOCK"
    assert "RISK_PORTFOLIO_HARD_BLOCK" in result.gates_added
    assert "PROTECTIVE_EXIT_OVERRIDE" not in result.gates_added


def test_hard_block_with_a_soft_throttle_still_holds_a_full_close() -> None:
    # The compiler never sets both for a full close with these budgets, so feed
    # the combination directly: a hard block always holds the exit.
    kwargs = _real_approval_kwargs(
        10,
        -10,
        _risk_policy(),
        approve_execution_candidate_func=lambda policy, candidate: ExecutionApproval(
            False, 0, 5.0, ("RISK_SYMBOL_SOFT_THROTTLE", "RISK_PORTFOLIO_HARD_BLOCK", "ZERO_QTY")
        ),
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason == "EXECUTION_APPROVAL_BLOCK"
    assert "CLOSING_EXIT_OVERRIDE" not in result.gates_added


def _paper_floor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    monkeypatch.setenv("AI_TRADING_PAPER_SAMPLING_SOFT_THROTTLE_FLOOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_PAPER_SAMPLING_MIN_EXPECTED_NET_EDGE_BPS", "0.10")


def test_paper_floor_does_not_shrink_a_full_short_cover(monkeypatch: pytest.MonkeyPatch) -> None:
    _paper_floor_env(monkeypatch)
    # Cover a 100-share short with portfolio gross far above its soft limit: the
    # throttle rounds it to 0; the paper floor used to send 1 share before the
    # exit override could act.
    kwargs = _real_approval_kwargs(
        -100, 100, _risk_policy(portfolio_soft_gross_dollars=1000.0), portfolio_current_gross=1_000_000.0
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.delta_shares == 100
    assert "CLOSING_EXIT_OVERRIDE" in result.gates_added
    assert "PAPER_SAMPLE_SOFT_THROTTLE_FLOOR" not in result.gates_added


@pytest.mark.parametrize(("current", "delta"), [(100, 4), (-100, 4)], ids=["add", "partial_cover"])
def test_paper_floor_still_samples_one_share_of_a_throttled_add_or_partial_cover(
    monkeypatch: pytest.MonkeyPatch, current: int, delta: int
) -> None:
    _paper_floor_env(monkeypatch)
    # The symbol soft limit rounds a 4-share buy to 0; the floor sends one
    # share, exactly as on main, for an add and for a partial cover.
    kwargs = _real_approval_kwargs(current, delta, _risk_policy(symbol_soft_dollars=100.0))

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.delta_shares == 1
    assert "PAPER_SAMPLE_SOFT_THROTTLE_FLOOR" in result.gates_added
    assert "CLOSING_EXIT_OVERRIDE" not in result.gates_added


def test_overridable_block_passes_a_protective_exit_whatever_else_the_approval_lists() -> None:
    # As on main: when the approval is held only by overridable blocks, a
    # protective exit goes out at full size even next to another reason.
    kwargs = _real_approval_kwargs(
        10,
        -3,
        _risk_policy(),
        protective_exit=True,
        approve_execution_candidate_func=lambda policy, candidate: ExecutionApproval(
            False, 0, -3.0, ("NET_EDGE_FLOOR_BLOCK", "SOME_OTHER_REASON")
        ),
    )

    result = prepare_netting_symbol_approval(**cast(Any, kwargs))

    assert result.blocked_reason is None
    assert result.delta_shares == -3
    assert "PROTECTIVE_EXIT_OVERRIDE" in result.gates_added
