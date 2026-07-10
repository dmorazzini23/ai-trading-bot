from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

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


def test_contextual_live_cost_can_only_raise_fixed_cost_and_reports_source() -> None:
    now = datetime.now(UTC)
    live_cost_model = {
        "generated_at": now.isoformat(),
        "by_symbol_side_session_order_type_volatility": [
            {
                "symbol": "AAPL",
                "side": "buy",
                "session_regime": "midday",
                "order_type": "limit",
                "volatility_bucket": "normal",
                "sample_count": 10,
                "sufficient_samples": True,
                "p90_total_cost_bps": 20.0,
                "last_observed_at": now.isoformat(),
            }
        ],
    }
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
    params = {
        "transaction_cost_bps": 4.0,
        "slippage_bps": 2.0,
        "allow_short": True,
    }

    fixed = simulate_executed_trades(
        y_true=[0.01, 0.01],
        y_pred=[1.0, 1.0],
        params=params,
    )
    contextual = simulate_executed_trades(
        y_true=[0.01, 0.01],
        y_pred=[1.0, 1.0],
        params=params,
        execution_context=context,
        live_cost_model=live_cost_model,
    )
    repeated = simulate_executed_trades(
        y_true=[0.01, 0.01],
        y_pred=[1.0, 1.0],
        params=params,
        execution_context=context,
        live_cost_model=live_cost_model,
    )

    assert repeated == contextual
    assert contextual["cost_return"] == pytest.approx(0.002)
    assert contextual["cost_return"] > fixed["cost_return"]
    assert contextual["mean_applied_cost_bps"] == pytest.approx(20.0)
    assert contextual["cost_source_live_count"] == 1.0
    assert contextual["cost_source_fallback_count"] == 0.0


def test_contextual_missing_cost_uses_positive_fallback_on_turnover_only() -> None:
    contextual = simulate_executed_trades(
        y_true=[0.01, 0.01],
        y_pred=[1.0, 1.0],
        params={
            "transaction_cost_bps": 0.0,
            "slippage_bps": 0.0,
            "allow_short": True,
        },
        execution_context=[{}, {}],
    )
    neutral = simulate_executed_trades(
        y_true=[0.01, 0.01],
        y_pred=[0.0, 0.0],
        params={
            "transaction_cost_bps": 0.0,
            "slippage_bps": 0.0,
            "allow_short": True,
        },
        execution_context=[{}, {}],
    )

    assert contextual["cost_return"] == pytest.approx(0.0001)
    assert contextual["mean_applied_cost_bps"] == pytest.approx(1.0)
    assert contextual["cost_source_fallback_count"] == 1.0
    assert neutral["cost_return"] == 0.0
    assert neutral["cost_source_fallback_count"] == 0.0
