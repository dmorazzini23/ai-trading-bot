from __future__ import annotations

from datetime import UTC, datetime

import pytest

import ai_trading.execution.liquidity as liquidity_module
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


def test_analyze_volume_patterns_and_insufficient_data() -> None:
    analyzer = LiquidityAnalyzer()
    out = analyzer._analyze_volume_patterns(
        {"volume": [100] * 10 + [100] * 9 + [400]}
    )
    assert out["data_points"] == 20
    assert out["volume_pattern"] == "surging"
    assert out["volume_ratio"] > 1.5
    assert out["volume_trend"] > 0.1

    insufficient = analyzer._analyze_volume_patterns({"volume": [1, 2, 3]})
    assert "error" in insufficient


def test_analyze_bid_ask_spread_with_direct_and_estimated_paths() -> None:
    analyzer = LiquidityAnalyzer()

    direct = analyzer._analyze_bid_ask_spread(
        {"bid": [99.0, 100.0], "ask": [101.0, 102.0]},
        current_price=100.0,
    )
    assert direct["data_points"] == 2
    assert direct["spread_quality"] == "poor"
    assert direct["spread_basis_points"] > 30

    estimated = analyzer._analyze_bid_ask_spread(
        {"close": [100.0, 101.0, 99.0, 100.0, 102.0, 98.0, 100.0, 101.0, 99.0, 100.0]},
        current_price=100.0,
    )
    assert estimated["spread_quality"] == "estimated"
    assert estimated["estimated_spread"] > 0


def test_analyze_dollar_volume_paths() -> None:
    analyzer = LiquidityAnalyzer()

    valid = analyzer._analyze_dollar_volume(
        {"volume": [100, 200, 300, 400, 500], "close": [10, 10, 10, 10, 10]}
    )
    assert valid["avg_dollar_volume"] == pytest.approx(3000)
    assert valid["current_dollar_volume"] == 5000
    assert valid["data_points"] == 5

    insufficient = analyzer._analyze_dollar_volume(
        {"volume": [100, 200], "close": [10, 10]}
    )
    assert "error" in insufficient

    no_valid = analyzer._analyze_dollar_volume(
        {"volume": [0, 0, 0, 0, 0], "close": [10, 10, 10, 10, 10]}
    )
    assert "error" in no_valid


def test_analyze_market_hours_liquidity_with_time_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = LiquidityAnalyzer()

    class _FakeDateTime:
        current = datetime(2026, 1, 5, 8, 15, tzinfo=UTC)  # Monday pre-market

        @classmethod
        def now(cls, tz=None):
            _ = tz
            return cls.current

    monkeypatch.setattr(liquidity_module, "datetime", _FakeDateTime)
    premarket = analyzer._analyze_market_hours_liquidity()
    assert premarket["market_session"] == MarketHours.PRE_MARKET
    assert premarket["liquidity_impact"] == "reduced"
    assert premarket["is_monday"] is True

    _FakeDateTime.current = datetime(2026, 1, 16, 15, 45, tzinfo=UTC)  # Friday close hour
    friday = analyzer._analyze_market_hours_liquidity()
    assert friday["is_friday"] is True
    assert friday["liquidity_impact"] == "low"

    _FakeDateTime.current = datetime(2026, 1, 17, 12, 0, tzinfo=UTC)  # Saturday
    weekend = analyzer._analyze_market_hours_liquidity()
    assert weekend["market_session"] == MarketHours.AFTER_HOURS
    assert weekend["is_weekend"] is True


@pytest.mark.parametrize(
    ("ratio", "trend", "expected"),
    [
        (1.6, 0.2, "surging"),
        (1.3, 0.0, "elevated"),
        (0.4, -0.2, "declining"),
        (0.7, 0.0, "below_average"),
        (1.0, 0.0, "normal"),
    ],
)
def test_classify_volume_pattern_variants(
    ratio: float,
    trend: float,
    expected: str,
) -> None:
    analyzer = LiquidityAnalyzer()
    assert analyzer._classify_volume_pattern(ratio, trend) == expected


def test_analyze_liquidity_end_to_end_and_history_trim() -> None:
    analyzer = LiquidityAnalyzer()
    market_data = {
        "volume": [1_000_000 + i * 10_000 for i in range(30)],
        "close": [100.0 + i * 0.1 for i in range(30)],
        "bid": [99.8 + i * 0.1 for i in range(30)],
        "ask": [100.2 + i * 0.1 for i in range(30)],
    }

    out = analyzer.analyze_liquidity("AAPL", market_data, current_price=103.0)
    assert out["symbol"] == "AAPL"
    assert out["liquidity_level"] in {
        LiquidityLevel.VERY_LOW,
        LiquidityLevel.LOW,
        LiquidityLevel.NORMAL,
        LiquidityLevel.HIGH,
        LiquidityLevel.VERY_HIGH,
    }
    assert "execution_recommendations" in out
    assert "AAPL" in analyzer.liquidity_history

    base = {
        "timestamp": datetime.now(UTC),
        "liquidity_level": LiquidityLevel.NORMAL,
        "dollar_volume_analysis": {"avg_dollar_volume": 1_000_000},
        "volume_analysis": {"volume_ratio": 1.0},
    }
    for i in range(60):
        entry = {**base, "timestamp": datetime.now(UTC)}
        analyzer._update_liquidity_history("TRIM", entry)
        _ = i
    assert len(analyzer.liquidity_history["TRIM"]) == 50


def test_get_liquidity_trend_and_manager_empty_paths() -> None:
    analyzer = LiquidityAnalyzer()
    err = analyzer.get_liquidity_trend("NONE")
    assert err["error"] == "Insufficient history"

    analyzer.liquidity_history["AAPL"] = [
        {
            "timestamp": datetime.now(UTC),
            "liquidity_level": LiquidityLevel.LOW,
            "avg_dollar_volume": 500_000,
            "volume_ratio": 0.8,
        },
        {
            "timestamp": datetime.now(UTC),
            "liquidity_level": LiquidityLevel.HIGH,
            "avg_dollar_volume": 2_000_000,
            "volume_ratio": 1.2,
        },
    ]
    trend = analyzer.get_liquidity_trend("AAPL")
    assert trend["trend_direction"] == "improving"
    assert trend["current_level"] == LiquidityLevel.HIGH

    manager = LiquidityManager()
    assert manager.get_portfolio_liquidity_summary()["error"] == "No liquidity data available"
    assert manager.update_symbol_liquidity("AAPL", None)["error"] == "No market data"
