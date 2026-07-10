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


def test_walkforward_defaults_to_long_only_when_short_policy_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AI_TRADING_WALK_FORWARD_ALLOW_SHORT", raising=False)
    monkeypatch.delenv("TRADING__ALLOW_SHORTS", raising=False)

    evaluator = WalkForwardEvaluator(output_dir=str(tmp_path))

    assert evaluator.trade_simulation_params["allow_short"] is False


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
    assert aggregate["turnover_annual"] == 2.5 / 5 * 252
    assert aggregate["executed_trades_annual"] == 3 / 5 * 252


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
        y_pred=y_pred.tolist(),
        params=evaluator.trade_simulation_params,
    )

    assert wrapped == direct


def test_walkforward_fold_sim_threads_contextual_costs(tmp_path) -> None:
    evaluator = WalkForwardEvaluator(output_dir=str(tmp_path))
    evaluator.trade_simulation_params = {
        "signal_threshold": 0.0,
        "transaction_cost_bps": 3.0,
        "slippage_bps": 2.0,
        "allow_short": True,
        "max_abs_position": 1.0,
    }
    now = datetime.now(UTC)
    context = [
        {
            "symbol": "AAPL",
            "session_regime": "midday",
            "order_type": "limit",
            "volatility_bucket": "normal",
        },
        {
            "symbol": "AAPL",
            "session_regime": "midday",
            "order_type": "limit",
            "volatility_bucket": "normal",
        },
    ]
    live_cost_model = {
        "generated_at": now.isoformat(),
        "by_symbol_side_session_order_type_volatility": [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": 8,
                "sufficient_samples": True,
                "p90_total_cost_bps": 15.0,
                "last_observed_at": now.isoformat(),
            }
        ],
    }
    y_true = pd.Series([0.01, 0.02], dtype=float)
    y_pred = np.array([0.5, 0.5], dtype=float)

    wrapped = evaluator._simulate_fold_trades(
        y_true=y_true,
        y_pred=y_pred,
        execution_context=context,
        live_cost_model=live_cost_model,
    )
    direct = simulate_executed_trades(
        y_true=y_true.values,
        y_pred=y_pred.tolist(),
        params=evaluator.trade_simulation_params,
        execution_context=context,
        live_cost_model=live_cost_model,
    )

    assert wrapped == direct
    assert wrapped["mean_applied_cost_bps"] == 15.0
    assert wrapped["cost_source_live_count"] == 1.0
