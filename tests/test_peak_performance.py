#!/usr/bin/env python3.12
"""
Tests for peak performance hardening modules.
"""

from datetime import UTC, datetime

import numpy as np
import pytest


# Test idempotency


def test_position_reconciliation():
    """Test position reconciliation logic."""

    from ai_trading.core.interfaces import Position
    from ai_trading.execution.reconcile import PositionReconciler

    reconciler = PositionReconciler(tolerance_pct=0.01)

    # Create test positions
    local_positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=100,
            market_value=15000.0,
            cost_basis=15000.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(UTC),
        ),
        "GOOGL": Position(
            symbol="GOOGL",
            quantity=50,
            market_value=140000.0,
            cost_basis=140000.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(UTC),
        ),
    }

    broker_positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=102,
            market_value=15300.0,
            cost_basis=15300.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(UTC),
        ),  # Small drift
        "MSFT": Position(
            symbol="MSFT",
            quantity=75,
            market_value=22500.0,
            cost_basis=22500.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(UTC),
        ),  # Missing locally
    }

    # Test drift detection
    drifts = reconciler.reconcile_positions(local_positions, broker_positions)

    assert len(drifts) >= 2  # AAPL drift + GOOGL missing + MSFT missing

    # Check AAPL drift
    aapl_drift = next((d for d in drifts if d.symbol == "AAPL"), None)
    assert aapl_drift is not None
    assert aapl_drift.drift_qty == -2.0  # local - broker




def test_symbol_costs():
    """Test symbol-aware cost model."""
    from ai_trading.execution.costs import SymbolCostModel, SymbolCosts

    # Test cost calculation
    costs = SymbolCosts(
        symbol="TEST",
        half_spread_bps=2.0,
        slip_k=1.5,
        commission_bps=0.5,
        min_commission=float(np.float32(2.0)),
    )

    # Test slippage calculation
    slippage = costs.slippage_cost_bps(volume_ratio=2.0)
    expected_slippage = 1.5 * np.sqrt(2.0)
    assert abs(slippage - expected_slippage) < 0.01

    # Test total cost
    total_cost = costs.total_execution_cost_bps(volume_ratio=1.5)
    expected_total = (2.0 * 2) + 0.5 + (1.5 * np.sqrt(1.5))  # spread + commission + slippage
    assert abs(total_cost - expected_total) < 0.01

    # Test cost model
    model = SymbolCostModel()

    # Get costs for new symbol
    symbol_costs = model.get_costs("NEWTEST")
    assert symbol_costs.symbol == "NEWTEST"
    assert symbol_costs.half_spread_bps > 0

    # Test cost impact calculation
    impact = model.calculate_position_impact(
        "NEWTEST",
        position_value=float(np.int64(10000)),
        volume_ratio=float(np.float32(1.0)),
    )
    assert "cost_bps" in impact
    assert "cost_dollars" in impact
    assert impact["cost_dollars"] > 0










def test_smart_order_routing():
    """Test smart order routing functionality."""
    from ai_trading.execution.order_policy import (
        MarketData,
        OrderUrgency,
        SmartOrderRouter,
    )

    router = SmartOrderRouter()

    # Create test market data
    market_data = MarketData(symbol="TEST", bid=100.0, ask=100.1, mid=100.05, spread_bps=10.0, volume_ratio=1.0)

    # Test limit price calculation
    limit_price, order_type = router.calculate_limit_price(market_data, side="buy", urgency=OrderUrgency.MEDIUM)

    assert isinstance(limit_price, float)
    assert limit_price > market_data.bid  # Buy should be above bid
    assert limit_price <= market_data.mid  # But not above mid for marketable limit

    # Test order request creation
    order_request = router.create_order_request(
        symbol="TEST", side="buy", quantity=100, market_data=market_data, urgency=OrderUrgency.MEDIUM
    )

    assert order_request["symbol"] == "TEST"
    assert order_request["side"] == "buy"
    assert order_request["quantity"] == 100
    assert "cost_estimate" in order_request
    assert "cost_bps" in order_request["cost_estimate"]


# Smoke test for backtester with cost enforcement
def test_backtest_cost_enforcement():
    """Test that backtester respects cost model."""
    # This would be a more complex integration test
    # For now, just ensure the cost model can be imported and used

    from ai_trading.execution.costs import get_cost_model

    model = get_cost_model()

    # Test cost adjustment
    adjusted_size, cost_info = model.adjust_position_size(
        symbol="TEST",
        target_size=float(np.int64(10000)),
        max_cost_bps=float(np.float32(15.0)),
        volume_ratio=float(np.int64(1)),
    )

    assert isinstance(adjusted_size, (int, float, np.floating))
    assert isinstance(cost_info, dict)

    # If costs are within limit, size should be unchanged
    if float(cost_info.get("cost_bps", 0)) <= 15.0:
        assert adjusted_size == 10000.0
    else:
        assert adjusted_size < 10000.0  # Should be scaled down


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
