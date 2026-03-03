from __future__ import annotations

import pandas as pd

from ai_trading.research.institutional_validation import (
    compute_risk_adjusted_scorecard,
    run_monte_carlo_trade_sequence_stress,
    run_purged_walk_forward_validation,
    run_regime_split_validation,
)


def test_monte_carlo_trade_sequence_stress_is_deterministic() -> None:
    returns = [0.01, -0.004, 0.003, -0.002, 0.005]
    first = run_monte_carlo_trade_sequence_stress(returns, trials=200, seed=7)
    second = run_monte_carlo_trade_sequence_stress(returns, trials=200, seed=7)

    assert first["p05_return_bps"] == second["p05_return_bps"]
    assert first["p95_max_drawdown_bps"] == second["p95_max_drawdown_bps"]


def test_purged_walk_forward_validation_runs() -> None:
    ts = pd.date_range("2024-01-01", periods=260, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "ret": [0.001 if idx % 3 else -0.0008 for idx in range(len(ts))],
        }
    )
    report = run_purged_walk_forward_validation(
        frame,
        return_col="ret",
        timestamp_col="timestamp",
        n_splits=4,
        embargo_pct=0.02,
        purge_pct=0.01,
        min_fold_samples=15,
    )

    assert report["fold_count"] >= 1
    assert 0.0 <= report["pass_ratio"] <= 1.0


def test_regime_split_validation_reports_pass_ratio() -> None:
    frame = pd.DataFrame(
        {
            "regime": ["trend"] * 40 + ["chop"] * 40 + ["high_vol"] * 40,
            "ret": [0.001] * 40 + [0.0003] * 40 + [-0.0002] * 40,
        }
    )
    report = run_regime_split_validation(
        frame,
        regime_col="regime",
        return_col="ret",
        min_samples=20,
        min_expectancy_bps=0.0,
        min_hit_rate=0.45,
    )

    assert report["eligible_regimes"] == 3
    assert 0.0 <= report["pass_ratio"] <= 1.0
    assert "trend" in report["regimes"]


def test_compute_risk_adjusted_scorecard_contains_institutional_metrics() -> None:
    scorecard = compute_risk_adjusted_scorecard([0.01, -0.004, 0.003, -0.002, 0.005])

    assert "sharpe_ratio" in scorecard
    assert "sortino_ratio" in scorecard
    assert "calmar_ratio" in scorecard
    assert "tail_loss_95" in scorecard
    assert "risk_of_ruin" in scorecard
