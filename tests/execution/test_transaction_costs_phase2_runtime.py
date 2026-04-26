from __future__ import annotations

import math

import pytest

from ai_trading.execution.transaction_costs import (
    LiquidityTier,
    TransactionCostBreakdown,
    TransactionCostCalculator,
    TradeType,
    create_transaction_cost_calculator,
    estimate_cost,
)


def _market_data() -> dict[str, object]:
    return {
        "prices": {"AAPL": 100.0, "THIN": 5.0},
        "quotes": {"AAPL": {"bid": 99.95, "ask": 100.05}},
        "volumes": {"AAPL": 2_000_000, "MID": 250_000, "LOW": 50_000, "THIN": 2_000},
        "volatility": {"AAPL": 0.04},
    }


def test_validation_and_breakdown_normalization() -> None:
    assert estimate_cost(10, 50, bps=2.0) == pytest.approx(0.10)

    breakdown = TransactionCostBreakdown(
        spread_cost=1.0,
        commission=2.0,
        market_impact=3.0,
        opportunity_cost=4.0,
        borrowing_cost=5.0,
        total_cost=999.0,
        cost_per_share=0.1,
        cost_percentage=2.0,
    )

    assert breakdown.total_cost == 15.0
    assert breakdown.cost_percentage == 1.0

    with pytest.raises(ValueError, match="commission_rate_gt1"):
        TransactionCostCalculator(commission_rate=1.5)
    with pytest.raises(ValueError, match="min_commission_invalid"):
        TransactionCostCalculator(min_commission=math.nan)
    with pytest.raises(ValueError, match="safety_margin_multiplier_invalid"):
        TransactionCostCalculator(safety_margin_multiplier=0.0)


def test_spread_estimation_and_liquidity_branches() -> None:
    calculator = TransactionCostCalculator(commission_rate=0.001, max_commission=25.0)
    data = _market_data()

    assert calculator.calculate_spread_cost("AAPL", 100, data) == pytest.approx(5.0)
    assert calculator.calculate_spread_cost("AAPL", -100, {"prices": {"AAPL": 100.0}, "quotes": {"AAPL": "stale"}}) > 0.0
    assert calculator.calculate_spread_cost("AAPL", 100, {"prices": {"AAPL": 100.0}, "quotes": {"AAPL": {"bid": 0, "ask": 0}}}) > 0.0
    assert calculator._classify_liquidity("AAPL", data) is LiquidityTier.HIGH_LIQUIDITY  # noqa: SLF001
    assert calculator._classify_liquidity("MID", data) is LiquidityTier.MEDIUM_LIQUIDITY  # noqa: SLF001
    assert calculator._classify_liquidity("LOW", data) is LiquidityTier.LOW_LIQUIDITY  # noqa: SLF001
    assert calculator._classify_liquidity("THIN", data) is LiquidityTier.ILLIQUID  # noqa: SLF001
    assert calculator._estimate_spread_percentage("THIN", data) == pytest.approx(0.01)  # noqa: SLF001


def test_total_cost_market_limit_short_and_profitability() -> None:
    calculator = TransactionCostCalculator(
        commission_rate=0.001,
        min_commission=0.25,
        max_commission=25.0,
    )
    data = _market_data()

    market_cost = calculator.calculate_total_transaction_cost(
        "AAPL",
        trade_size=100,
        trade_type=TradeType.MARKET_ORDER,
        market_data=data,
        expected_delay=10.0,
        expected_return=0.02,
    )
    limit_short_cost = calculator.calculate_total_transaction_cost(
        "AAPL",
        trade_size=-100,
        trade_type=TradeType.LIMIT_ORDER,
        market_data=data,
        expected_delay=10.0,
        expected_return=0.02,
        holding_period_days=3.0,
    )

    assert market_cost.total_cost > market_cost.commission
    assert market_cost.opportunity_cost < limit_short_cost.opportunity_cost
    assert limit_short_cost.borrowing_cost > 0.0
    assert limit_short_cost.spread_cost == pytest.approx(2.5)

    profitable = calculator.validate_trade_profitability(
        "AAPL",
        trade_size=100,
        expected_profit=1_000.0,
        market_data=data,
        trade_type=TradeType.MARKET_ORDER,
        confidence_level=0.9,
    )
    low_confidence = calculator.validate_trade_profitability(
        "AAPL",
        trade_size=100,
        expected_profit=1_000.0,
        market_data=data,
        confidence_level=0.5,
    )
    zero_profit = calculator.validate_trade_profitability(
        "AAPL",
        trade_size=100,
        expected_profit=0.0,
        market_data=data,
    )

    assert profitable.is_profitable is True
    assert profitable.net_expected_profit > 0.0
    assert low_confidence.is_profitable is False
    assert zero_profit.is_profitable is False
    assert zero_profit.cost_ratio == float("inf")


def test_fallback_paths_and_factory_configuration() -> None:
    calculator = create_transaction_cost_calculator(
        {
            "commission_rate": 0.002,
            "min_commission": 0.5,
            "max_commission": 50.0,
            "safety_margin_multiplier": 4.0,
        }
    )

    assert calculator.commission_rate == pytest.approx(0.002)
    assert calculator.calculate_commission("AAPL", 10, 100.0) == pytest.approx(0.5)
    assert calculator.calculate_market_impact("AAPL", 10, {"prices": {"AAPL": 100.0}, "volumes": {"AAPL": "bad"}}) == pytest.approx((3.5, 1.5))
    assert calculator.calculate_opportunity_cost("AAPL", "bad", 0.01, 1_000.0) == 0.0  # type: ignore[arg-type]
    assert calculator.calculate_borrowing_cost("AAPL", -10, 1_000.0, "bad") == 0.0  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid trade parameters"):
        calculator.validate_trade_profitability(
            "AAPL",
            trade_size=0.0,
            expected_profit=100.0,
            market_data=_market_data(),
        )
