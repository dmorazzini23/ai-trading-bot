from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

from ai_trading.core.netting_symbol_approval import prepare_netting_symbol_approval


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
    result = prepare_netting_symbol_approval(**cast(Any, _base_kwargs()))

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
