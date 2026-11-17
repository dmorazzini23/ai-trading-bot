from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_trading.execution import liquidity as liq


def test_liquidity_analysis_recommendations_trends_and_manager() -> None:
    analyzer = liq.LiquidityAnalyzer()
    data = {
        "volume": [100_000 * i for i in range(1, 25)],
        "close": [20.0 + i * 0.1 for i in range(24)],
        "bid": [20.0] * 24,
        "ask": [20.01] * 24,
    }

    result = analyzer.analyze_liquidity("AAPL", data, current_price=22.0)
    assert result["liquidity_level"] in {liq.LiquidityLevel.NORMAL, liq.LiquidityLevel.HIGH, liq.LiquidityLevel.VERY_HIGH}
    assert result["volume_analysis"]["volume_pattern"] in {"surging", "elevated", "normal"}
    assert "execution_strategy" in result["execution_recommendations"]

    analyzer.liquidity_history["AAPL"] = [
        {"avg_dollar_volume": 1_000_000.0, "volume_ratio": 0.8, "liquidity_level": liq.LiquidityLevel.LOW},
        {"avg_dollar_volume": 4_000_000.0, "volume_ratio": 1.2, "liquidity_level": liq.LiquidityLevel.HIGH},
    ]
    trend = analyzer.get_liquidity_trend("AAPL")
    assert trend["trend_direction"] == "improving"
    assert trend["dollar_volume_trend"] == 3.0

    assert analyzer.calculate_optimal_order_size(100, liq.LiquidityLevel.VERY_HIGH, 10_000)["execution_method"] == "single_order"
    sliced = analyzer.calculate_optimal_order_size(10_000, liq.LiquidityLevel.LOW, 10_000)
    assert sliced["execution_method"] == "sliced_orders"
    assert sliced["size_adjustment"] == "reduced_for_liquidity"

    manager = liq.LiquidityManager()
    manager.symbol_liquidity = {
        "AAPL": result,
        "THIN": {
            "liquidity_level": liq.LiquidityLevel.VERY_LOW,
            "dollar_volume_analysis": {"avg_dollar_volume": 10_000.0},
            "spread_analysis": {"spread_basis_points": 100.0},
            "timestamp": datetime.now(UTC),
        },
    }
    manager._update_portfolio_liquidity_score()
    summary = manager.get_portfolio_liquidity_summary()
    assert summary["total_symbols"] == 2
    assert manager.get_illiquid_positions()[0]["symbol"] == "THIN"
    with pytest.raises(ValueError, match="Missing symbol"):
        manager.pre_trade_check({}, data)


def test_liquidity_error_and_market_hour_branches(monkeypatch) -> None:
    analyzer = liq.LiquidityAnalyzer()
    assert analyzer.analyze_liquidity("BAD", None)["error"] == "No market data"
    assert analyzer._analyze_volume_patterns({"volume": [1, 2]})["error"] == "Insufficient volume data"
    assert analyzer._analyze_bid_ask_spread({"close": [1, 2]}, 2.0)["error"]
    assert analyzer._analyze_dollar_volume({"volume": [0, 0, 0, 0, 0], "close": [1, 1, 1, 1, 1]})["error"]
    assert analyzer._determine_liquidity_level({}, {}, {}) is liq.LiquidityLevel.VERY_LOW
    assert analyzer._classify_volume_pattern(0.4, -0.2) == "declining"
    assert analyzer._classify_volume_pattern(0.7, 0.0) == "below_average"
    assert analyzer._liquidity_level_to_numeric(liq.LiquidityLevel.VERY_HIGH) == 5

    class FakeDateTime:
        current = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)

        @classmethod
        def now(cls, _tz):
            return cls.current

    monkeypatch.setattr(liq, "datetime", FakeDateTime)
    assert analyzer._analyze_market_hours_liquidity()["market_session"] is liq.MarketHours.AFTER_HOURS
    FakeDateTime.current = datetime(2026, 4, 20, 9, 15, tzinfo=UTC)
    assert analyzer._analyze_market_hours_liquidity()["liquidity_impact"] in {"high", "reduced", "elevated"}


