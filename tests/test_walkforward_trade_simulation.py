from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from ai_trading.evaluation.execution_sim import simulate_executed_trades
from ai_trading.evaluation.walkforward import WalkForwardEvaluator


def test_simulate_fold_trades_applies_costs(tmp_path) -> None:
    evaluator = WalkForwardEvaluator(output_dir=str(tmp_path))
    evaluator.trade_simulation_params = {
        "signal_threshold": 0.0,
        "transaction_cost_bps": 8.0,
        "slippage_bps": 4.0,
        "allow_short": True,
        "max_abs_position": 1.0,
    }
    y_true = pd.Series([0.010, -0.015, 0.012, -0.006], dtype=float)
    y_pred = np.array([0.2, -0.4, 0.7, -0.3], dtype=float)

    metrics = evaluator._simulate_fold_trades(y_true=y_true, y_pred=y_pred)

    assert metrics["trade_count"] >= 1.0
    assert metrics["cost_return"] > 0.0
    assert metrics["net_return"] < metrics["gross_return"]


def test_simulate_fold_trades_respects_no_short_mode(tmp_path) -> None:
    evaluator = WalkForwardEvaluator(output_dir=str(tmp_path))
    evaluator.trade_simulation_params = {
        "signal_threshold": 0.0,
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
        "allow_short": False,
        "max_abs_position": 1.0,
    }
    y_true = pd.Series([-0.01, -0.02, -0.03], dtype=float)
    y_pred = np.array([-0.5, -0.6, -0.7], dtype=float)

    metrics = evaluator._simulate_fold_trades(y_true=y_true, y_pred=y_pred)

    assert metrics["signal_count"] == 0.0
    assert metrics["gross_return"] == 0.0
    assert metrics["net_return"] == 0.0


def test_aggregate_metrics_include_executed_trade_fields(tmp_path) -> None:
    evaluator = WalkForwardEvaluator(output_dir=str(tmp_path))
    base = datetime.now(UTC)
    evaluator.fold_results = [{"metrics": {"period_days": 5}}]
    evaluator.equity_curve = pd.Series([100.0, 101.5], index=[base, base + timedelta(days=5)])
    evaluator.drawdown_series = pd.Series([0.0, 0.01], index=evaluator.equity_curve.index)

    aggregate = evaluator._calculate_aggregate_metrics(
        predictions_all=[0.2, -0.1, 0.3, 0.05],
        actual_all=[0.01, -0.02, 0.03, 0.0],
        splits=[{"period_days": 5}],
        fold_trade_metrics=[
            {
                "net_return": 0.015,
                "trade_count": 3.0,
                "turnover_units": 2.5,
                "cost_return": 0.0012,
            }
        ],
    )

    assert "executed_total_return" in aggregate
    assert aggregate["executed_total_return"] > 0.0
    assert aggregate["executed_trade_count"] == 3
    assert aggregate["executed_turnover_units"] == 2.5


def test_walkforward_fold_sim_wrapper_matches_execution_sim(tmp_path) -> None:
    evaluator = WalkForwardEvaluator(output_dir=str(tmp_path))
    evaluator.trade_simulation_params = {
        "signal_threshold": 0.05,
        "transaction_cost_bps": 5.0,
        "slippage_bps": 2.0,
        "allow_short": True,
        "max_abs_position": 1.0,
    }
    y_true = pd.Series([0.01, -0.02, 0.03, 0.00], dtype=float)
    y_pred = np.array([0.2, -0.1, 0.3, 0.01], dtype=float)

    wrapped = evaluator._simulate_fold_trades(y_true=y_true, y_pred=y_pred)
    direct = simulate_executed_trades(
        y_true=y_true.values,
        y_pred=y_pred,
        params=evaluator.trade_simulation_params,
    )

    assert wrapped == direct
