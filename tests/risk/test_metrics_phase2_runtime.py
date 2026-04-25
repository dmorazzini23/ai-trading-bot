from __future__ import annotations

import math

import pytest

from ai_trading.risk.metrics import DrawdownAnalyzer, RiskMetricsCalculator


def _returns() -> list[float]:
    return [0.01, -0.02, 0.015, -0.03, 0.02, -0.01] * 10


def test_risk_metrics_var_expected_shortfall_and_tail_loss() -> None:
    calc = RiskMetricsCalculator()
    returns = _returns()

    assert calc.calculate_var([0.01] * 10) == 0.0
    assert calc.calculate_expected_shortfall([0.01] * 10) == 0.0
    assert calc.calculate_var(returns, 0.95) == pytest.approx(0.03)
    assert calc.calculate_expected_shortfall(returns, 0.95) == pytest.approx(0.03)
    assert calc.calculate_tail_loss(returns, 0.95) == pytest.approx(
        calc.calculate_expected_shortfall(returns, 0.95)
    )


def test_risk_metrics_ratios_drawdown_and_scorecard() -> None:
    calc = RiskMetricsCalculator()
    returns = _returns()

    assert calc.calculate_sharpe_ratio([0.01]) == 0.0
    assert calc.calculate_sharpe_ratio([0.01] * 5) == 0.0
    assert calc.calculate_sharpe_ratio(returns) != 0.0
    assert calc.calculate_sortino_ratio([0.01, 0.02, 0.03], risk_free_rate=0.0) == math.inf
    assert calc.calculate_sortino_ratio([-0.01, -0.02, 0.03]) != 0.0
    assert calc.calculate_max_drawdown([]) == 0.0
    assert calc.calculate_max_drawdown([0.1, -0.2, 0.05]) == pytest.approx(0.2)
    assert calc.calculate_calmar_ratio([0.01]) == 0.0
    assert calc.calculate_calmar_ratio(returns) != 0.0

    scorecard = calc.calculate_scorecard([0.01, float("nan"), -0.02, 0.03] * 10)

    assert set(scorecard) == {
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "tail_loss_95",
        "risk_of_ruin",
    }
    assert 0.0 <= scorecard["risk_of_ruin"] <= 1.0


def test_risk_of_ruin_branches() -> None:
    calc = RiskMetricsCalculator()

    assert calc.calculate_risk_of_ruin([0.01]) == 0.0
    assert calc.calculate_risk_of_ruin([0.01, 0.01]) == 0.0
    assert calc.calculate_risk_of_ruin([-0.01, -0.02, -0.03]) == 1.0
    assert calc.calculate_risk_of_ruin([0.02, -0.01, 0.03, -0.01]) < 1.0


def test_drawdown_analyzer_statistics_status_and_recovery() -> None:
    analyzer = DrawdownAnalyzer()
    values = [100.0, 120.0, 90.0, 95.0, 130.0, 100.0, 140.0]

    stats = analyzer.calculate_drawdowns(values)

    assert analyzer.calculate_drawdowns([]) == {}
    assert stats["max_drawdown"] == pytest.approx(0.25)
    assert stats["max_drawdown_start"] == 1
    assert stats["max_drawdown_end"] == 2
    assert stats["num_drawdown_periods"] == 2
    assert analyzer.is_in_drawdown(90.0, 100.0) == (True, 0.1)
    assert analyzer.is_in_drawdown(100.0, 100.0) == (False, 0.0)
    assert analyzer.is_in_drawdown(90.0, 0.0) == (False, 0.0)
    assert analyzer.calculate_recovery_time(values, 1, 2) == 2
    assert analyzer.calculate_recovery_time(values, 4, 5) == 1
    assert analyzer.calculate_recovery_time(values, 10, 11) is None
