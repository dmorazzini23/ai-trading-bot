"""Tests for narrowed exceptions in risk metrics."""

from __future__ import annotations

from ai_trading.risk.metrics import RiskMetricsCalculator


def test_var_handles_bad_values_gracefully() -> None:
    """Invalid returns should return 0.0 instead of raising."""  # AI-AGENT-REF: narrow exception test
    rmc = RiskMetricsCalculator()
    returns = ["bad", None, {}]  # type: ignore
    assert rmc.calculate_var(returns, confidence_level=0.95) == 0.0


def test_institutional_scorecard_metrics_present() -> None:
    rmc = RiskMetricsCalculator()
    returns = [0.01, -0.005, 0.004, 0.002, -0.003, 0.006, 0.001, -0.002]
    scorecard = rmc.calculate_scorecard(returns)

    assert set(scorecard) == {
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "tail_loss_95",
        "risk_of_ruin",
    }
    assert scorecard["max_drawdown"] >= 0.0
    assert 0.0 <= scorecard["risk_of_ruin"] <= 1.0
