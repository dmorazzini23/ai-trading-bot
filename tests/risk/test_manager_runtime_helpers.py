from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest

from ai_trading.core.enums import RiskLevel
from ai_trading.risk import manager as rm


def _disable_portfolio_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rm,
        "get_settings",
        lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=False),
    )


def _enable_portfolio_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rm,
        "get_settings",
        lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True),
    )


def test_risk_manager_trade_assessment_clips_oversized_position(monkeypatch):
    _disable_portfolio_features(monkeypatch)
    manager = rm.RiskManager(RiskLevel.CONSERVATIVE)

    assessment = manager.assess_trade_risk(
        "SPY",
        quantity=100,
        price=100.0,
        portfolio_value=10_000.0,
        position_history=[],
    )

    assert assessment["approved"] is False
    assert assessment["recommended_size"] == 2
    assert assessment["metrics"]["position_size_pct"] == pytest.approx(1.0)
    assert assessment["warnings"]


def test_risk_manager_trade_assessment_approves_small_trade(monkeypatch):
    _disable_portfolio_features(monkeypatch)
    manager = rm.RiskManager(RiskLevel.AGGRESSIVE)

    assessment = manager.assess_trade_risk(
        "SPY",
        quantity=2,
        price=100.0,
        portfolio_value=10_000.0,
        position_history=[],
    )

    assert assessment["approved"] is True
    assert assessment["recommended_size"] == 2
    assert assessment["risk_score"] < 70


def test_risk_manager_kelly_features_reduce_recommended_size(monkeypatch):
    _enable_portfolio_features(monkeypatch)

    class KellyCriterion:
        def calculate_from_returns(self, returns):
            assert returns == [0.01, -0.02]
            return 0.01, {"sample_size": len(returns)}

    class KellyCalculator:
        def __init__(self):
            self.kelly_criterion = KellyCriterion()

    monkeypatch.setattr(rm, "KellyCalculator", KellyCalculator)
    manager = rm.RiskManager(RiskLevel.AGGRESSIVE)

    assessment = manager.assess_trade_risk(
        "SPY",
        quantity=20,
        price=100.0,
        portfolio_value=10_000.0,
        position_history=[{"return": 0.01}, {"return": -0.02}],
    )

    assert assessment["recommended_size"] == 1
    assert assessment["metrics"]["kelly_fraction"] == 0.01
    assert any("Kelly criterion" in warning for warning in assessment["warnings"])


def test_risk_manager_trade_assessment_error_path(monkeypatch):
    _disable_portfolio_features(monkeypatch)
    manager = rm.RiskManager()

    assessment = manager.assess_trade_risk(
        "SPY",
        quantity=1,
        price="bad",  # type: ignore[arg-type]
        portfolio_value=10_000.0,
        position_history=[],
    )

    assert assessment["approved"] is False
    assert assessment["risk_score"] == 100.0
    assert assessment["recommended_size"] == 0


def test_check_portfolio_risk_disabled_and_empty(monkeypatch):
    _disable_portfolio_features(monkeypatch)
    manager = rm.RiskManager()
    disabled = manager.check_portfolio_risk([], {})
    assert disabled["overall_risk_level"] == "Unknown"

    _enable_portfolio_features(monkeypatch)
    manager = rm.RiskManager()
    empty = manager.check_portfolio_risk([], {})
    assert empty["overall_risk_level"] == "Low"


def test_check_portfolio_risk_flags_concentration_and_drawdown(monkeypatch):
    _enable_portfolio_features(monkeypatch)
    manager = rm.RiskManager(RiskLevel.CONSERVATIVE)
    manager.current_drawdown = 0.10
    positions = [
        {"symbol": "A", "market_value": 9_000.0, "sector": "Tech"},
        {"symbol": "B", "market_value": 1_000.0, "sector": "Utility"},
    ]

    assessment = manager.check_portfolio_risk(positions, {})

    assert assessment["overall_risk_level"] in {"High", "Critical"}
    assert assessment["metrics"]["max_position_concentration"] == pytest.approx(0.9)
    assert assessment["metrics"]["max_sector_concentration"] == pytest.approx(0.9)
    assert len(assessment["alerts"]) >= 3
    assert "Reduce position sizes" in assessment["recommendations"]
    assert manager.current_portfolio_risk == pytest.approx(assessment["risk_score"] / 100)


def test_check_portfolio_risk_uses_gross_for_long_short_books(monkeypatch):
    _enable_portfolio_features(monkeypatch)
    manager = rm.RiskManager(RiskLevel.AGGRESSIVE)

    assessment = manager.check_portfolio_risk(
        [
            {"symbol": "LONG", "market_value": 100.0, "sector": "Tech"},
            {"symbol": "SHORT", "market_value": -100.0, "sector": "Tech"},
        ],
        {},
    )

    assert assessment["metrics"]["max_position_concentration"] == pytest.approx(0.5)
    assert assessment["metrics"]["max_sector_concentration"] == pytest.approx(1.0)


def test_check_portfolio_risk_error_path(monkeypatch):
    _enable_portfolio_features(monkeypatch)
    manager = rm.RiskManager()

    assessment = manager.check_portfolio_risk(
        [{"market_value": "bad", "sector": "Tech"}],
        {},
    )

    assert assessment["overall_risk_level"] == "Critical"
    assert assessment["risk_score"] == 100.0
    assert assessment["recommendations"] == ["Manual review required"]


def test_drawdown_and_alert_lifecycle(monkeypatch):
    _disable_portfolio_features(monkeypatch)
    manager = rm.RiskManager()

    manager.update_drawdown(current_value=80.0, peak_value=100.0)
    assert manager.current_drawdown == pytest.approx(0.2)
    manager.update_drawdown(current_value=120.0, peak_value=100.0)
    assert manager.current_drawdown == 0.0
    manager.update_drawdown(current_value=50.0, peak_value=0.0)
    assert manager.current_drawdown == 0.0

    manager.risk_alerts.append(
        {
            "timestamp": datetime.now(UTC) - timedelta(days=2),
            "type": "old",
            "message": "old",
            "severity": "low",
        }
    )
    manager.add_risk_alert("drawdown", "risk rose", "high")
    alerts = manager.get_risk_alerts()
    assert len(alerts) == 1
    assert alerts[0]["type"] == "drawdown"
    alerts.append({"type": "mutated"})
    assert len(manager.risk_alerts) == 1


def test_portfolio_var_and_expected_shortfall_edges():
    assessor = rm.PortfolioRiskAssessor()
    returns = [-0.10, -0.05, -0.03, -0.02, -0.01] + [0.01] * 35

    assert assessor.calculate_var([0.01] * 10) == 0.0
    assert assessor.calculate_var(returns, confidence_level=0.95) == pytest.approx(0.03)
    assert assessor.calculate_expected_shortfall([0.01] * 10) == 0.0
    assert assessor.calculate_expected_shortfall(returns, confidence_level=0.95) == pytest.approx(
        0.075
    )
    assert assessor.calculate_expected_shortfall(returns, confidence_level=0.999) == pytest.approx(
        assessor.calculate_var(returns, confidence_level=0.999)
    )


def test_portfolio_var_error_paths():
    assessor = rm.PortfolioRiskAssessor()

    assert assessor.calculate_var([object()] * 30) == 0.0  # type: ignore[list-item]
    assert assessor.calculate_expected_shortfall([object()] * 30) == 0.0  # type: ignore[list-item]


def test_correlation_matrix_pairs_and_constant_series():
    assessor = rm.PortfolioRiskAssessor()
    ascending = [float(i) for i in range(40)]
    descending = [float(40 - i) for i in range(40)]
    constant = [1.0] * 40

    correlations = assessor.calculate_correlation_matrix(
        {"A": ascending, "B": descending, "C": constant, "SHORT": [1.0] * 10}
    )

    assert correlations["A_B"] == pytest.approx(-1.0)
    assert correlations["B_A"] == pytest.approx(-1.0)
    assert correlations["A_C"] == 0.0
    assert "A_SHORT" not in correlations
    assert assessor._calculate_correlation([1.0], [1.0]) == 0.0
    assert assessor._calculate_correlation([1.0, 2.0], [1.0]) == 0.0


def test_correlation_matrix_error_path():
    assessor = rm.PortfolioRiskAssessor()

    bad_returns = cast(list[float], [object()] * 40)
    assert assessor.calculate_correlation_matrix({"A": bad_returns, "B": [1.0] * 40}) == {
        "A_B": 0.0,
        "B_A": 0.0,
    }


def test_stress_test_portfolio_applies_market_and_symbol_shocks():
    assessor = rm.PortfolioRiskAssessor()
    positions = [
        {"symbol": "A", "market_value": 100.0},
        {"symbol": "B", "market_value": 200.0},
    ]

    results = assessor.stress_test_portfolio(
        positions,
        {
            "market_down": {"market_shock": -0.10},
            "mixed": {"A": 0.10, "market_shock": -0.05},
        },
    )

    assert results["scenario_count"] == 2
    assert results["scenarios"]["market_down"]["portfolio_value_after"] == pytest.approx(270.0)
    assert results["scenarios"]["mixed"]["portfolio_change_pct"] == pytest.approx(0.0)
    assert results["worst_case_loss"] == pytest.approx(-0.10)
    assert results["best_case_gain"] == pytest.approx(0.0)


def test_stress_test_portfolio_uses_signed_pnl_over_gross_exposure():
    assessor = rm.PortfolioRiskAssessor()

    results = assessor.stress_test_portfolio(
        [
            {"symbol": "LONG", "market_value": 100.0},
            {"symbol": "SHORT", "market_value": -100.0},
        ],
        {"market_down": {"market_shock": -0.10}},
    )

    assert results["scenario_count"] == 1
    assert results["scenarios"]["market_down"]["portfolio_value_before"] == pytest.approx(200.0)
    assert results["scenarios"]["market_down"]["portfolio_change_pct"] == pytest.approx(0.0)


def test_stress_test_zero_and_error_paths():
    assessor = rm.PortfolioRiskAssessor()

    zero = assessor.stress_test_portfolio([{"market_value": 0.0}], {"down": {"market_shock": -1}})
    assert zero["scenario_count"] == 0
    assert assessor._apply_stress_scenario([{"market_value": 0.0}], {"market_shock": -1})[
        "portfolio_change_pct"
    ] == 0
    assert assessor.stress_test_portfolio([{"market_value": object()}], {"bad": {}})[
        "error"
    ]
