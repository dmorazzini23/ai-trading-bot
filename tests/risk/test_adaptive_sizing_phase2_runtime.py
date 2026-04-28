from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_trading.core.enums import RiskLevel
from ai_trading.risk import adaptive_sizing as adaptive


def test_market_condition_analyzer_classifies_regimes() -> None:
    analyzer = adaptive.MarketConditionAnalyzer(lookback_days=30)
    analyzer.vol_thresholds = {
        "extremely_low": -0.1,
        "low": 0.2,
        "high": 1.1,
        "extremely_high": 1.2,
    }

    bull_prices = [100.0 + i for i in range(40)]
    bear_prices = [140.0 - i for i in range(40)]
    flat_prices = [100.0 + ((-1) ** i) * 0.1 for i in range(40)]

    assert analyzer.analyze_market_regime({"SPY": bull_prices}) is adaptive.MarketRegime.BULL_TRENDING
    assert analyzer.analyze_market_regime({"SPY": bear_prices}) is adaptive.MarketRegime.BEAR_TRENDING
    assert analyzer.analyze_market_regime({"SPY": flat_prices}) is adaptive.MarketRegime.SIDEWAYS_RANGE
    assert analyzer.analyze_market_regime({}) is adaptive.MarketRegime.NORMAL
    assert analyzer.analyze_market_regime({"SPY": [1.0, 2.0]}) is adaptive.MarketRegime.NORMAL


def test_volatility_regime_percentiles_and_correlation_matrix() -> None:
    analyzer = adaptive.MarketConditionAnalyzer(lookback_days=60)
    low_then_high = [0.001, -0.001] * 20 + [0.05, -0.05] * 20
    high_then_low = [0.05, -0.05] * 20 + [0.001, -0.001] * 20

    assert analyzer.assess_volatility_regime([0.01] * 10) is adaptive.VolatilityRegime.NORMAL
    assert analyzer.assess_volatility_regime(low_then_high) in {
        adaptive.VolatilityRegime.HIGH,
        adaptive.VolatilityRegime.EXTREMELY_HIGH,
    }
    assert analyzer.assess_volatility_regime(high_then_low) in {
        adaptive.VolatilityRegime.LOW,
        adaptive.VolatilityRegime.EXTREMELY_LOW,
    }

    correlations = analyzer.calculate_correlation_matrix(
        {
            "AAPL": [0.01, -0.01] * 20,
            "MSFT": [0.01, -0.01] * 20,
            "TLT": [-0.01, 0.01] * 20,
        }
    )
    assert correlations["AAPL_MSFT"] == pytest.approx(1.0)
    assert correlations["AAPL_TLT"] == pytest.approx(-1.0)
    assert correlations["MSFT_AAPL"] == correlations["AAPL_MSFT"]


def test_market_condition_helpers_handle_degenerate_inputs() -> None:
    analyzer = adaptive.MarketConditionAnalyzer()

    assert analyzer._get_market_proxy({"XYZ": [1.0]}) == "XYZ"
    assert analyzer._get_market_proxy({}) is None
    assert analyzer._calculate_trend_strength([1.0] * 5) == 0.0
    assert analyzer._calculate_rolling_volatility([1.0]) == 0.0
    assert analyzer._calculate_volatility_percentile(0.1, [1.0] * 5) == 0.5
    assert analyzer._calculate_correlation([1.0], [1.0]) == 0.0
    assert analyzer._calculate_correlation([1.0, 1.0], [2.0, 2.0]) == 0.0
    assert analyzer._get_percentile_rank(2.0, [1.0, 2.0, 2.0, 3.0]) == pytest.approx(0.5)


def test_adaptive_position_combines_base_size_with_market_penalties(monkeypatch: pytest.MonkeyPatch) -> None:
    sizer = adaptive.AdaptivePositionSizer(RiskLevel.MODERATE)
    monkeypatch.setattr(
        sizer.dynamic_sizer,
        "calculate_optimal_position",
        lambda *_args, **_kwargs: {"recommended_size": 100, "warnings": []},
    )
    monkeypatch.setattr(
        sizer.market_analyzer,
        "analyze_market_regime",
        lambda _price_data: adaptive.MarketRegime.HIGH_VOLATILITY,
    )
    monkeypatch.setattr(
        sizer.market_analyzer,
        "assess_volatility_regime",
        lambda _returns: adaptive.VolatilityRegime.EXTREMELY_HIGH,
    )
    monkeypatch.setattr(
        sizer.market_analyzer,
        "calculate_correlation_matrix",
        lambda _returns_data: {"SPY_AAPL": 0.9},
    )

    result = sizer.calculate_adaptive_position(
        "SPY",
        account_equity=100_000.0,
        entry_price=100.0,
        market_data={"atr": 2.0},
        portfolio_data={
            "price_data": {"SPY": [100.0] * 60},
            "returns_data": {"SPY": [0.01] * 60, "AAPL": [0.01] * 60},
            "current_positions": {"AAPL": {"notional_value": 10_000.0}},
        },
    )

    assert result["base_calculation"]["recommended_size"] == 100
    assert result["market_adjustments"]["market_regime"] == "high_volatility"
    assert result["market_adjustments"]["volatility_regime"] == "extremely_high"
    assert result["market_adjustments"]["correlation_penalty"] == pytest.approx(0.5)
    assert result["recommended_size"] == 10
    assert result["warnings"] == [
        "High volatility regime - reduced position sizing",
        "Extremely high volatility - consider smaller positions",
        "High correlation with existing positions - size reduced",
    ]
    assert "high volatility environment" in result["metadata"]["sizing_rationale"]


def test_adaptive_position_reapplies_caps_after_positive_multipliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizer = adaptive.AdaptivePositionSizer(RiskLevel.MODERATE)
    monkeypatch.setattr(
        sizer.dynamic_sizer,
        "calculate_optimal_position",
        lambda *_args, **_kwargs: {
            "recommended_size": 100,
            "sizing_methods": {"concentration_limit": 50},
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        sizer.market_analyzer,
        "analyze_market_regime",
        lambda _price_data: adaptive.MarketRegime.BULL_TRENDING,
    )
    monkeypatch.setattr(
        sizer.market_analyzer,
        "assess_volatility_regime",
        lambda _returns: adaptive.VolatilityRegime.EXTREMELY_LOW,
    )
    monkeypatch.setattr(
        sizer.market_analyzer,
        "calculate_correlation_matrix",
        lambda _returns_data: {},
    )

    result = sizer.calculate_adaptive_position(
        "SPY",
        account_equity=100_000.0,
        entry_price=100.0,
        market_data={"atr": 2.0},
        portfolio_data={"price_data": {"SPY": [100.0] * 60}, "returns_data": {"SPY": [0.0] * 60}},
    )

    assert result["recommended_size"] == 50
    assert "Position size capped by concentration limit" in result["warnings"]


def test_adaptive_position_returns_base_warnings_when_base_size_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizer = adaptive.AdaptivePositionSizer()
    monkeypatch.setattr(
        sizer.dynamic_sizer,
        "calculate_optimal_position",
        lambda *_args, **_kwargs: {
            "recommended_size": 0,
            "warnings": ["No valid position size calculated"],
        },
    )

    result = sizer.calculate_adaptive_position("SPY", 100_000.0, 100.0, {}, {})

    assert result["recommended_size"] == 0
    assert result["warnings"] == ["No valid position size calculated"]


def test_adaptive_limits_risk_assessment_and_rationale() -> None:
    sizer = adaptive.AdaptivePositionSizer()

    crisis = sizer.get_regime_based_limits(adaptive.MarketRegime.CRISIS)
    high_vol = sizer.get_regime_based_limits(adaptive.MarketRegime.HIGH_VOLATILITY)
    bull = sizer.get_regime_based_limits(adaptive.MarketRegime.BULL_TRENDING)
    normal = sizer.get_regime_based_limits(adaptive.MarketRegime.NORMAL)

    assert crisis["max_position_pct"] < normal["max_position_pct"]
    assert high_vol["max_portfolio_risk"] < normal["max_portfolio_risk"]
    assert bull["max_position_pct"] > normal["max_position_pct"]
    assert sizer._calculate_correlation_penalty("SPY", {}, {}) == 0.0
    assert sizer._calculate_correlation_penalty(
        "SPY",
        {"SPY_AAPL": 0.4, "SPY_MSFT": 0.8},
        {
            "current_positions": {
                "AAPL": {"notional_value": 5_000.0},
                "MSFT": {"notional_value": 5_000.0},
            }
        },
    ) == pytest.approx(0.5)
    assert sizer._calculate_correlation_penalty(
        "SPY",
        {"SPY_AAPL": 0.6, "SPY_MSFT": 0.6},
        {
            "current_positions": {
                "AAPL": {"notional_value": 10_000.0},
                "MSFT": {"notional_value": -10_000.0},
            }
        },
    ) == pytest.approx(0.5)
    assert sizer._assess_position_risk(0, 100.0, 100_000.0, {}) == {
        "error": "Invalid inputs for risk assessment"
    }
    risk = sizer._assess_position_risk(100, 100.0, 100_000.0, {"atr": 5.0})
    assert risk["risk_level"] == "normal"
    assert risk["position_percentage"] == 0.1
    assert sizer._generate_sizing_rationale(1.2, 1.3, 0.4) == (
        "Position adjusted for: favorable market regime, low volatility environment, "
        "high correlation with existing positions"
    )
    assert sizer._generate_sizing_rationale(1.0, 1.0, 0.0) == "Standard position sizing applied"
