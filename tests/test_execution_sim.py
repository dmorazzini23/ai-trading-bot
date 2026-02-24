from __future__ import annotations

import numpy as np

from ai_trading.evaluation.execution_sim import simulate_executed_trades


def test_simulate_executed_trades_applies_costs() -> None:
    y_true = np.array([0.010, -0.015, 0.012, -0.006], dtype=float)
    y_pred = np.array([0.2, -0.4, 0.7, -0.3], dtype=float)
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
    y_true = np.array([-0.01, -0.02, -0.03], dtype=float)
    y_pred = np.array([-0.5, -0.6, -0.7], dtype=float)
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
