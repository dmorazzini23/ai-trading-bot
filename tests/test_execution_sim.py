from __future__ import annotations

import numpy as np

from ai_trading.evaluation.execution_sim import simulate_executed_trades
from ai_trading.evaluation.execution_sim import ExecutionSimConfig


def test_execution_sim_defaults_are_not_frictionless() -> None:
    cfg = ExecutionSimConfig.from_mapping({})

    assert cfg.transaction_cost_bps > 0.0
    assert cfg.slippage_bps > 0.0


def test_execution_sim_parses_string_false_for_short_mode() -> None:
    cfg = ExecutionSimConfig.from_mapping({"allow_short": "false"})

    assert cfg.allow_short is False


def test_simulate_executed_trades_applies_costs() -> None:
    y_true = np.array([0.010, -0.015, 0.012, -0.006], dtype=float).tolist()
    y_pred = np.array([0.2, -0.4, 0.7, -0.3], dtype=float).tolist()
    metrics = simulate_executed_trades(
        y_true=y_true,
        y_pred=y_pred,
        params={
            "signal_threshold": 0.0,
            "transaction_cost_bps": 8.0,
            "slippage_bps": 4.0,
            "allow_short": True,
            "max_abs_position": 1.0,
        },
    )

    assert metrics["trade_count"] >= 1.0
    assert metrics["cost_return"] > 0.0
    assert metrics["net_return"] < metrics["gross_return"]


def test_simulate_executed_trades_respects_no_short_mode() -> None:
    y_true = np.array([-0.01, -0.02, -0.03], dtype=float).tolist()
    y_pred = np.array([-0.5, -0.6, -0.7], dtype=float).tolist()
    metrics = simulate_executed_trades(
        y_true=y_true,
        y_pred=y_pred,
        params={
            "signal_threshold": 0.0,
            "transaction_cost_bps": 0.0,
            "slippage_bps": 0.0,
            "allow_short": False,
            "max_abs_position": 1.0,
        },
    )

    assert metrics["signal_count"] == 0.0
    assert metrics["gross_return"] == 0.0
    assert metrics["net_return"] == 0.0


def test_simulate_executed_trades_keeps_non_finite_inputs_neutral() -> None:
    metrics = simulate_executed_trades(
        y_true=[0.01, float("nan"), 0.02, float("inf")],
        y_pred=[0.2, 0.8, float("nan"), float("-inf")],
        params={
            "signal_threshold": 0.0,
            "transaction_cost_bps": 0.0,
            "slippage_bps": 0.0,
            "allow_short": True,
            "max_abs_position": 1.0,
        },
    )

    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["gross_return"] == 0.01
    assert np.isclose(metrics["net_return"], 0.01)
    assert metrics["signal_count"] == 1.0


def test_execution_sim_config_rejects_non_finite_params() -> None:
    cfg = ExecutionSimConfig.from_mapping(
        {
            "signal_threshold": float("nan"),
            "transaction_cost_bps": float("inf"),
            "slippage_bps": float("-inf"),
            "max_abs_position": float("nan"),
        }
    )

    assert cfg.signal_threshold == 0.0
    assert cfg.transaction_cost_bps == 1.0
    assert cfg.slippage_bps == 5.0
    assert cfg.max_abs_position == 1.0


def test_execution_sim_direct_config_rejects_non_finite_params() -> None:
    cfg = ExecutionSimConfig(
        signal_threshold=float("nan"),
        transaction_cost_bps=float("inf"),
        slippage_bps=float("-inf"),
        max_abs_position=float("nan"),
    )

    assert cfg.signal_threshold == 0.0
    assert cfg.transaction_cost_bps == 1.0
    assert cfg.slippage_bps == 5.0
    assert cfg.max_abs_position == 1.0
