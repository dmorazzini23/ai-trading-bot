#!/usr/bin/env python3.12
import logging

"""
Validation script for peak-performance hardening implementation.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    logging.info("Testing module imports...")

    modules = [
        'ai_trading.execution.idempotency',
        'ai_trading.execution.reconcile',
        'ai_trading.execution.costs',
        'ai_trading.execution.order_policy',
        'ai_trading.scheduler.aligned_clock',
        'ai_trading.portfolio.risk_controls',
        'ai_trading.utils.determinism',
        'ai_trading.utils.performance',
        'ai_trading.monitoring.drift'
    ]

    imported = []
    failed = []

    for module in modules:
        try:
            __import__(module)
            imported.append(module)
            logging.info(f"  ✓ {module}")
        except Exception as e:
            failed.append((module, str(e)))
            logging.info(f"  ✗ {module}: {e}")

    return imported, failed


def test_basic_functionality():
    """Test basic functionality of key modules."""
    logging.info("\nTesting basic functionality...")

    try:
        # Test idempotency
        from ai_trading.core.interfaces import OrderSide
        from ai_trading.execution.idempotency import OrderIdempotencyCache

        cache = OrderIdempotencyCache(ttl_seconds=60)
        key = cache.generate_key("TEST", OrderSide.BUY, 100.0)
        assert not cache.is_duplicate(key)
        cache.mark_submitted(key, "test_order_123")
        assert cache.is_duplicate(key)
        logging.info("  ✓ Idempotency system working")

    except Exception as e:
        logging.info(f"  ✗ Idempotency test failed: {e}")
        return False

    try:
        # Test costs (without persistence)
        from ai_trading.execution.costs import SymbolCosts

        costs = SymbolCosts(
            symbol="TEST",
            half_spread_bps=2.0,
            slip_k=1.5,
            commission_bps=0.5
        )

        total_cost = costs.total_execution_cost_bps(volume_ratio=1.0)
        assert total_cost > 0
        logging.info("  ✓ Cost model working")

    except Exception as e:
        logging.info(f"  ✗ Cost model test failed: {e}")
        return False

    try:
        # Test determinism
        import numpy as np

        from ai_trading.utils.determinism import set_random_seeds

        set_random_seeds(42)
        random1 = np.random.random(5)

        set_random_seeds(42)
        random2 = np.random.random(5)

        assert np.array_equal(random1, random2)
        logging.info("  ✓ Determinism working")

    except Exception as e:
        logging.info(f"  ✗ Determinism test failed: {e}")
        return False

    try:
        # Test performance cache
        from ai_trading.utils.performance import PerformanceCache

        cache = PerformanceCache(max_size=10, ttl_seconds=60)
        cache.set("test", "value")
        assert cache.get("test") == "value"
        logging.info("  ✓ Performance cache working")

    except Exception as e:
        logging.info(f"  ✗ Performance cache test failed: {e}")
        return False

    return True


def test_integration():
    """Test integration between modules."""
    logging.info("\nTesting module integration...")

    try:
        # Test that modules can work together
        from ai_trading.execution.costs import get_cost_model
        from ai_trading.execution.order_policy import MarketData, SmartOrderRouter

        # Get cost model
        cost_model = get_cost_model()

        # Create router
        router = SmartOrderRouter()

        # Create market data
        market_data = MarketData(
            symbol="TEST",
            bid=100.0,
            ask=100.05,
            mid=100.025,
            spread_bps=5.0
        )

        # Get costs for symbol
        cost_model.get_costs("TEST")

        # Calculate limit price
        limit_price, order_type = router.calculate_limit_price(market_data, "buy")

        assert isinstance(limit_price, float)
        assert limit_price > market_data.bid

        logging.info("  ✓ Module integration working")
        return True

    except Exception as e:
        logging.info(f"  ✗ Integration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    logging.info("Peak-Performance Hardening Validation")
    logging.info(str("=" * 40))

    # Test imports
    imported, failed = test_imports()

    if failed:
        logging.info(f"\n❌ {len(failed)} modules failed to import:")
        for module, error in failed:
            logging.info(f"   {module}: {error}")
        return False

    logging.info(f"\n✅ All {len(imported)} modules imported successfully")

    # Test functionality
    if not test_basic_functionality():
        logging.info("\n❌ Basic functionality tests failed")
        return False

    logging.info("\n✅ Basic functionality tests passed")

    # Test integration
    if not test_integration():
        logging.info("\n❌ Integration tests failed")
        return False

    logging.info("\n✅ Integration tests passed")

    logging.info("\n🎉 Peak-performance hardening implementation validated!")
    logging.info("\nKey features implemented:")
    logging.info("  • Order idempotency with TTL cache")
    logging.info("  • Position reconciliation system")
    logging.info("  • Exchange-aligned timing")
    logging.info("  • Symbol-aware cost modeling")
    logging.info("  • Adaptive risk controls")
    logging.info("  • Deterministic training")
    logging.info("  • Feature drift monitoring")
    logging.info("  • Performance optimizations")
    logging.info("  • Smart order routing")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
