from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_trading.core.enums import OrderType
from ai_trading.execution.liquidity import (
    LiquidityAnalyzer,
    LiquidityLevel,
    LiquidityManager,
    MarketHours,
)


def test_determine_liquidity_level_very_high() -> None:
    analyzer = LiquidityAnalyzer()
    level = analyzer._determine_liquidity_level(
        volume_analysis={"volume_ratio": 2.1},
        dollar_volume_analysis={"avg_dollar_volume": analyzer.liquidity_thresholds["very_high"]},
        spread_analysis={"spread_basis_points": 2.0},
    )
    assert level == LiquidityLevel.VERY_HIGH


def test_generate_execution_recommendations_adjusts_for_session_and_volume_trend() -> None:
    analyzer = LiquidityAnalyzer()
    rec = analyzer._generate_execution_recommendations(
        liquidity_level=LiquidityLevel.NORMAL,
        volume_analysis={"volume_pattern": "declining"},
        market_hours_analysis={
            "market_session": MarketHours.AFTER_HOURS,
            "liquidity_impact": "very_low",
        },
    )

    assert rec["recommended_order_type"] == OrderType.LIMIT
    assert rec["execution_strategy"] == "patient"
    assert rec["max_participation_rate"] == pytest.approx(0.04)
    assert "Extended hours trading - reduced liquidity" in rec["risk_warnings"]
    assert "Declining volume trend" in rec["risk_warnings"]
    assert "Consider waiting for regular market hours" in rec["timing_recommendations"]


def test_calculate_optimal_order_size_single_order_when_within_liquidity_limit() -> None:
    analyzer = LiquidityAnalyzer()
    result = analyzer.calculate_optimal_order_size(
        target_size=100,
        liquidity_level=LiquidityLevel.HIGH,
        avg_volume=20_000,
    )
    assert result == {
        "recommended_size": 100,
        "execution_method": "single_order",
        "estimated_slices": 1,
        "size_adjustment": "none",
    }


def test_calculate_optimal_order_size_slices_when_target_too_large() -> None:
    analyzer = LiquidityAnalyzer()
    result = analyzer.calculate_optimal_order_size(
        target_size=1_000,
        liquidity_level=LiquidityLevel.LOW,
        avg_volume=2_000,
    )
    assert result["recommended_size"] == 100
    assert result["execution_method"] == "sliced_orders"
    assert result["estimated_slices"] == 10
    assert result["reduction_ratio"] == pytest.approx(0.1)


def test_liquidity_manager_summary_and_illiquid_positions() -> None:
    manager = LiquidityManager()
    now = datetime.now(UTC)
    manager.symbol_liquidity = {
        "AAA": {
            "liquidity_level": LiquidityLevel.VERY_HIGH,
            "dollar_volume_analysis": {"avg_dollar_volume": 80_000_000},
            "spread_analysis": {"spread_basis_points": 3.0},
            "timestamp": now,
        },
        "BBB": {
            "liquidity_level": LiquidityLevel.HIGH,
            "dollar_volume_analysis": {"avg_dollar_volume": 15_000_000},
            "spread_analysis": {"spread_basis_points": 8.0},
            "timestamp": now,
        },
        "CCC": {
            "liquidity_level": LiquidityLevel.LOW,
            "dollar_volume_analysis": {"avg_dollar_volume": 400_000},
            "spread_analysis": {"spread_basis_points": 40.0},
            "timestamp": now,
        },
    }
    manager._update_portfolio_liquidity_score()

    summary = manager.get_portfolio_liquidity_summary()
    assert summary["total_symbols"] == 3
    assert summary["portfolio_assessment"] == "excellent"
    assert summary["high_liquidity_percentage"] == pytest.approx(66.6666666667)

    illiquid = manager.get_illiquid_positions(threshold=LiquidityLevel.LOW)
    assert [item["symbol"] for item in illiquid] == ["CCC"]

