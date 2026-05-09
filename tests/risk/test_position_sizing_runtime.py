from __future__ import annotations

import pytest

from ai_trading.core.enums import RiskLevel
from ai_trading.risk.position_sizing import (
    ATRPositionSizer,
    DynamicPositionSizer,
    PortfolioPositionManager,
    VolatilityPositionSizer,
)


def test_atr_position_size_and_stop_levels_cover_long_short_and_invalid_inputs() -> None:
    sizer = ATRPositionSizer(risk_per_trade=0.02)

    assert sizer.calculate_position_size(100_000.0, 100.0, 2.0) == 500
    assert sizer.calculate_position_size(100_000.0, 100.0, 2.0, stop_distance_multiplier=4.0) == 250
    assert sizer.calculate_position_size(0.0, 100.0, 2.0) == 0
    assert sizer.calculate_stop_levels(100.0, 2.0, side="long") == {
        "stop_loss": 96.0,
        "take_profit": 105.0,
    }
    assert sizer.calculate_stop_levels(100.0, 2.0, side="short") == {
        "stop_loss": 104.0,
        "take_profit": 95.0,
    }
    assert sizer.calculate_stop_levels(0.0, 2.0) == {"stop_loss": 0.0, "take_profit": 0.0}


def test_volatility_multiplier_bounds_and_adjusted_size() -> None:
    sizer = VolatilityPositionSizer(target_volatility=0.15)

    assert sizer.calculate_volatility_multiplier([0.01] * 9) == 1.0
    assert sizer.calculate_volatility_multiplier([0.0] * 20) == 1.0
    assert sizer.calculate_volatility_multiplier([0.05, -0.05] * 10) == 0.2
    assert sizer.calculate_volatility_multiplier([0.001, -0.001] * 10) == 2.0
    assert sizer.calculate_position_size(100, [0.01] * 9) == 0
    assert sizer.calculate_position_size(100, [0.001, -0.001] * 10) == 200


def test_dynamic_position_size_fails_closed_without_atr_evidence() -> None:
    sizer = DynamicPositionSizer(RiskLevel.CONSERVATIVE)

    result = sizer.calculate_optimal_position(
        symbol="AAPL",
        account_equity=100_000.0,
        entry_price=100.0,
        market_data={"atr": 0.0},
        historical_data={},
    )

    assert result["recommended_size"] == 0
    assert result["sizing_methods"] == {}
    assert result["warnings"] == ["Invalid ATR value"]


def test_dynamic_position_size_fails_closed_without_return_evidence() -> None:
    sizer = DynamicPositionSizer(RiskLevel.CONSERVATIVE)

    result = sizer.calculate_optimal_position(
        symbol="AAPL",
        account_equity=100_000.0,
        entry_price=100.0,
        market_data={"atr": 2.0},
        historical_data={},
    )

    assert result["recommended_size"] == 0
    assert result["sizing_methods"]["atr_based"] == 625
    assert result["warnings"] == ["Insufficient return data for volatility adjustment"]


def test_dynamic_position_size_records_positive_kelly_method() -> None:
    sizer = DynamicPositionSizer(RiskLevel.AGGRESSIVE)
    returns = [0.01, -0.005] * 10
    trade_history = [{"return": 0.04} for _ in range(20)] + [{"return": -0.01} for _ in range(10)]

    result = sizer.calculate_optimal_position(
        symbol="MSFT",
        account_equity=100_000.0,
        entry_price=100.0,
        market_data={"atr": 2.0},
        historical_data={"returns": returns, "trade_history": trade_history},
    )

    assert result["recommended_size"] > 0
    assert "kelly_based" in result["sizing_methods"]
    assert result["risk_metrics"]["kelly_fraction"] > 0
    assert result["risk_metrics"]["kelly_stats"]["total_trades"] == 30


def test_dynamic_position_size_error_path_and_large_position_warning() -> None:
    sizer = DynamicPositionSizer(RiskLevel.AGGRESSIVE)

    error_result = sizer.calculate_optimal_position(
        symbol="BAD",
        account_equity=100_000.0,
        entry_price=0.0,
        market_data={"atr": 1.0},
        historical_data={"returns": [0.01, -0.01] * 10},
    )
    large_result = sizer.calculate_optimal_position(
        symbol="BIG",
        account_equity=100_000.0,
        entry_price=100.0,
        market_data={"atr": 0.5},
        historical_data={"returns": [0.001, -0.001] * 10},
    )

    assert error_result["recommended_size"] == 0
    assert error_result["warnings"] == ["Invalid account equity or entry price"]
    assert large_result["recommended_size"] == 100
    assert "Large position size relative to account" in large_result["warnings"]


def test_dynamic_scaling_orders_and_risk_estimate() -> None:
    sizer = DynamicPositionSizer()

    assert sizer.calculate_scaling_orders(0, 100.0) == []
    orders = sizer.calculate_scaling_orders(100, 100.0, scaling_levels=3)

    assert [order["size"] for order in orders] == [50, 30, 20]
    assert [order["level"] for order in orders] == [1, 2, 3]
    assert sum(order["percentage"] for order in orders) == 100.0
    assert sizer._estimate_position_risk(0.0, 2.0) == 0.0
    assert sizer._estimate_position_risk(1_000.0, 2.0) == pytest.approx(0.4)


def test_portfolio_position_manager_assesses_updates_and_summarizes_positions() -> None:
    manager = PortfolioPositionManager(max_portfolio_risk=0.05)

    approved = manager.assess_new_position("AAPL", proposed_size=10, entry_price=100.0, account_equity=100_000.0)
    assert approved["approved"] is True
    assert approved["adjusted_size"] == 10

    oversized = manager.assess_new_position(
        "TSLA",
        proposed_size=1_000,
        entry_price=100.0,
        account_equity=100_000.0,
    )
    assert oversized["approved"] is False
    assert oversized["adjusted_size"] == 4
    assert oversized["recommendations"] == ["Consider splitting order across multiple sessions"]

    manager.update_position("AAPL", size=10, entry_price=100.0)
    manager.update_position("MSFT", size=5, entry_price=200.0)
    summary = manager.get_portfolio_summary()

    assert summary["position_count"] == 2
    assert summary["total_notional_value"] == 2_000.0
    assert summary["largest_position"] == 1_000.0

    manager.update_position("AAPL", size=0, entry_price=100.0)
    assert "AAPL" not in manager.get_portfolio_summary()["positions"]


def test_portfolio_position_manager_preserves_short_gross_accounting() -> None:
    manager = PortfolioPositionManager(max_portfolio_risk=1.0)

    assessment = manager.assess_new_position(
        "AAPL",
        proposed_size=-10,
        entry_price=100.0,
        account_equity=100_000.0,
    )
    assert assessment["approved"] is True
    assert assessment["adjusted_size"] == -10
    assert assessment["risk_impact"] == pytest.approx(0.005)

    manager.update_position("AAPL", size=-10, entry_price=100.0)
    summary = manager.get_portfolio_summary()

    assert summary["position_count"] == 1
    assert summary["total_notional_value"] == pytest.approx(1_000.0)
    assert summary["largest_position"] == pytest.approx(1_000.0)
    assert summary["total_risk_exposure"] == pytest.approx(20.0)
    assert summary["positions"]["AAPL"]["notional_value"] == pytest.approx(-1_000.0)


def test_portfolio_position_manager_risk_reduction_and_error_paths() -> None:
    manager = PortfolioPositionManager(max_portfolio_risk=0.01)
    manager.total_risk_exposure = 0.009

    reduced = manager.assess_new_position(
        "MSFT",
        proposed_size=100,
        entry_price=10.0,
        account_equity=100_000.0,
    )

    assert reduced["approved"] is False
    assert reduced["adjusted_size"] == 20
    assert reduced["risk_impact"] == pytest.approx(0.005)
    assert "Portfolio risk limit requires size reduction" in reduced["warnings"][0]
    assert reduced["recommendations"] == ["Consider splitting order across multiple sessions"]

    manager.current_positions = {"BAD": {"notional_value": "not-a-number"}}
    manager._recalculate_risk_exposure()
    assert manager.total_risk_exposure == 0.0
    assert "error" in manager.get_portfolio_summary()
