#!/usr/bin/env python3.12
"""
Tests for peak performance hardening modules.
"""

from datetime import UTC, datetime

import numpy as np
import pytest


# Test idempotency
def test_order_idempotency():
    """Test order idempotency caching."""
    # Clear any existing cache
    from ai_trading.execution.idempotency import (
        get_idempotency_cache,
        is_duplicate_order,
        mark_order_submitted,
    )
    cache = get_idempotency_cache()
    cache.clear_expired()

    symbol = "TEST"
    side = "buy"
    quantity = 100.0

    # First order should not be duplicate
    is_dup, order_id = is_duplicate_order(symbol, side, quantity)
    assert not is_dup
    assert order_id is None

    # Mark as submitted
    test_order_id = "order_123"
    mark_order_submitted(symbol, side, quantity, test_order_id)

    # Second identical order should be duplicate
    is_dup, existing_id = is_duplicate_order(symbol, side, quantity)
    assert is_dup
    assert existing_id == test_order_id


def test_position_reconciliation():
    """Test position reconciliation logic."""
    from datetime import datetime

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
            timestamp=datetime.now(UTC)
        ),
        "GOOGL": Position(
            symbol="GOOGL",
            quantity=50,
            market_value=140000.0,
            cost_basis=140000.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(UTC)
        )
    }

    broker_positions = {
        "AAPL": Position(
            symbol="AAPL",
            quantity=102,
            market_value=15300.0,
            cost_basis=15300.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(UTC)
        ),  # Small drift
        "MSFT": Position(
            symbol="MSFT",
            quantity=75,
            market_value=22500.0,
            cost_basis=22500.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(UTC)
        )    # Missing locally
    }

    # Test drift detection
    drifts = reconciler.reconcile_positions(local_positions, broker_positions)

    assert len(drifts) >= 2  # AAPL drift + GOOGL missing + MSFT missing

    # Check AAPL drift
    aapl_drift = next((d for d in drifts if d.symbol == "AAPL"), None)
    assert aapl_drift is not None
    assert aapl_drift.drift_qty == -2.0  # local - broker


def test_aligned_clock():
    """Test exchange-aligned clock functionality."""
    from ai_trading.scheduler.aligned_clock import AlignedClock

    clock = AlignedClock(max_skew_ms=250.0)

    # Test skew checking
    skew = clock.check_skew()
    assert isinstance(skew, float)
    assert abs(skew) < 5000  # Should be reasonable

    # Test bar validation
    validation = clock.ensure_final_bar("TEST", "1m")
    assert hasattr(validation, 'is_final')
    assert hasattr(validation, 'skew_ms')

    # Test market hours (basic check)
    test_time = datetime(2023, 6, 15, 14, 30, tzinfo=UTC)  # Weekday 2:30 PM UTC
    is_open = clock.is_market_open("AAPL", test_time)
    assert isinstance(is_open, bool)


def test_symbol_costs():
    """Test symbol-aware cost model."""
    from ai_trading.execution.costs import SymbolCostModel, SymbolCosts

    # Test cost calculation
    costs = SymbolCosts(
        symbol="TEST",
        half_spread_bps=2.0,
        slip_k=1.5,
        commission_bps=0.5
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
    impact = model.calculate_position_impact("NEWTEST", position_value=10000, volume_ratio=1.0)
    assert 'cost_bps' in impact
    assert 'cost_dollars' in impact
    assert impact['cost_dollars'] > 0


def test_adaptive_risk_controls():
    """Test adaptive risk control system."""
    from ai_trading.portfolio.risk_controls import AdaptiveRiskController
    pd = pytest.importorskip("pandas")

    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']

    returns_data = pd.DataFrame(
        np.random.normal(0, 0.02, (100, 3)),
        index=dates,
        columns=symbols
    )

    controller = AdaptiveRiskController()

    # Test volatility calculation
    vols = controller.calculate_volatilities(returns_data)
    assert len(vols) == 3
    for symbol in symbols:
        assert symbol in vols
        assert vols[symbol] > 0

    # Test correlation clustering (skip if scipy not available)
    try:
        clusters = controller.calculate_correlation_clusters(returns_data)
        assert len(clusters) == 3
        assert all(isinstance(cluster_id, int) for cluster_id in clusters.values())
    except ImportError:
        pass  # Skip if clustering dependencies not available

    # Test Kelly calculation
    expected_returns = {symbol: 0.01 for symbol in symbols}  # 1% expected return
    kelly_fractions = controller.calculate_kelly_fractions(expected_returns, vols)

    assert len(kelly_fractions) == 3
    for symbol in symbols:
        assert kelly_fractions[symbol] >= 0  # Kelly fractions should be non-negative


def test_determinism():
    """Test deterministic training setup."""
    from ai_trading.utils.determinism import hash_data, set_random_seeds
    pd = pytest.importorskip("pandas")

    # Test seed setting
    set_random_seeds(42)

    # Generate some random numbers to verify determinism
    np_random1 = np.random.random(5)

    # Reset and generate again
    set_random_seeds(42)
    np_random2 = np.random.random(5)

    np.testing.assert_array_equal(np_random1, np_random2)

    # Test data hashing
    test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    hash1 = hash_data(test_data)
    hash2 = hash_data(test_data)

    assert hash1 == hash2
    assert len(hash1) == 16  # Should be 16-char hash

    # Test different data produces different hash
    test_data2 = pd.DataFrame({'A': [1, 2, 4], 'B': [4, 5, 6]})  # Changed one value
    hash3 = hash_data(test_data2)

    assert hash1 != hash3


def test_drift_monitoring():
    """Test drift monitoring functionality."""
    from ai_trading.monitoring.drift import DriftMonitor
    pd = pytest.importorskip("pandas")

    monitor = DriftMonitor()

    # Create baseline and current features
    np.random.seed(42)
    baseline_features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000)
    })

    # Update baseline
    monitor.update_baseline(baseline_features)

    # Create current features with slight drift
    current_features = pd.DataFrame({
        'feature1': np.random.normal(0.1, 1, 500),  # Small mean shift
        'feature2': np.random.normal(0, 1.2, 500)   # Increased variance
    })

    # Monitor drift
    drift_metrics = monitor.monitor_feature_drift(current_features)

    assert len(drift_metrics) == 2
    for metric in drift_metrics:
        assert metric.psi_score >= 0
        assert metric.drift_level in ['low', 'medium', 'high']
        assert metric.sample_size == 500


def test_performance_optimizations():
    """Test performance optimization utilities."""
    from ai_trading.utils.performance import (
        PerformanceCache,
        VectorizedOperations,
        benchmark_operation,
    )
    pd = pytest.importorskip("pandas")

    # Test caching
    cache = PerformanceCache(max_size=10, ttl_seconds=60)

    cache.set("test_key", "test_value")
    assert cache.get("test_key") == "test_value"
    assert cache.get("nonexistent") is None

    # Test vectorized operations
    np.random.seed(42)
    prices = pd.Series(np.random.random(100).cumsum() + 100)

    # Test fast returns
    returns = VectorizedOperations.fast_returns(prices, periods=1)
    assert len(returns) == len(prices)
    assert np.isnan(returns.iloc[0])  # First value should be NaN

    # Test rolling z-score
    zscore = VectorizedOperations.rolling_zscore(prices, window=20)
    assert len(zscore) == len(prices)

    # Test benchmark function
    def dummy_operation(data):
        return data.sum()

    result = benchmark_operation("test_sum", dummy_operation, prices)
    assert result.operation == "test_sum"
    assert result.duration_ms > 0


def test_smart_order_routing():
    """Test smart order routing functionality."""
    from ai_trading.execution.order_policy import (
        MarketData,
        OrderUrgency,
        SmartOrderRouter,
    )

    router = SmartOrderRouter()

    # Create test market data
    market_data = MarketData(
        symbol="TEST",
        bid=100.0,
        ask=100.1,
        mid=100.05,
        spread_bps=10.0,
        volume_ratio=1.0
    )

    # Test limit price calculation
    limit_price, order_type = router.calculate_limit_price(
        market_data, side="buy", urgency=OrderUrgency.MEDIUM
    )

    assert isinstance(limit_price, float)
    assert limit_price > market_data.bid  # Buy should be above bid
    assert limit_price <= market_data.mid  # But not above mid for marketable limit

    # Test order request creation
    order_request = router.create_order_request(
        symbol="TEST",
        side="buy",
        quantity=100,
        market_data=market_data,
        urgency=OrderUrgency.MEDIUM
    )

    assert order_request['symbol'] == "TEST"
    assert order_request['side'] == "buy"
    assert order_request['quantity'] == 100
    assert 'cost_estimate' in order_request
    assert 'cost_bps' in order_request['cost_estimate']


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
        target_size=10000,
        max_cost_bps=15.0,
        volume_ratio=1.0
    )

    assert isinstance(adjusted_size, float)
    assert isinstance(cost_info, dict)

    # If costs are within limit, size should be unchanged
    if cost_info.get('cost_bps', 0) <= 15.0:
        assert adjusted_size == 10000
    else:
        assert adjusted_size < 10000  # Should be scaled down


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
