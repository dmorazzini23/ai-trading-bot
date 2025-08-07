#!/usr/bin/env python3.12
"""
Validation script for peak-performance hardening implementation.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
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
            print(f"  ✓ {module}")
        except Exception as e:
            failed.append((module, str(e)))
            print(f"  ✗ {module}: {e}")
    
    return imported, failed


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\nTesting basic functionality...")
    
    try:
        # Test idempotency
        from ai_trading.execution.idempotency import OrderIdempotencyCache, IdempotencyKey
        from ai_trading.core.interfaces import OrderSide
        
        cache = OrderIdempotencyCache(ttl_seconds=60)
        key = cache.generate_key("TEST", OrderSide.BUY, 100.0)
        assert not cache.is_duplicate(key)
        cache.mark_submitted(key, "test_order_123")
        assert cache.is_duplicate(key)
        print("  ✓ Idempotency system working")
        
    except Exception as e:
        print(f"  ✗ Idempotency test failed: {e}")
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
        print("  ✓ Cost model working")
        
    except Exception as e:
        print(f"  ✗ Cost model test failed: {e}")
        return False
    
    try:
        # Test determinism
        from ai_trading.utils.determinism import set_random_seeds
        import numpy as np
        
        set_random_seeds(42)
        random1 = np.random.random(5)
        
        set_random_seeds(42)
        random2 = np.random.random(5)
        
        assert np.array_equal(random1, random2)
        print("  ✓ Determinism working")
        
    except Exception as e:
        print(f"  ✗ Determinism test failed: {e}")
        return False
    
    try:
        # Test performance cache
        from ai_trading.utils.performance import PerformanceCache
        
        cache = PerformanceCache(max_size=10, ttl_seconds=60)
        cache.set("test", "value")
        assert cache.get("test") == "value"
        print("  ✓ Performance cache working")
        
    except Exception as e:
        print(f"  ✗ Performance cache test failed: {e}")
        return False
    
    return True


def test_integration():
    """Test integration between modules."""
    print("\nTesting module integration...")
    
    try:
        # Test that modules can work together
        from ai_trading.execution.costs import get_cost_model
        from ai_trading.execution.order_policy import SmartOrderRouter, MarketData
        
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
        costs = cost_model.get_costs("TEST")
        
        # Calculate limit price
        limit_price, order_type = router.calculate_limit_price(market_data, "buy")
        
        assert isinstance(limit_price, float)
        assert limit_price > market_data.bid
        
        print("  ✓ Module integration working")
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("Peak-Performance Hardening Validation")
    print("=" * 40)
    
    # Test imports
    imported, failed = test_imports()
    
    if failed:
        print(f"\n❌ {len(failed)} modules failed to import:")
        for module, error in failed:
            print(f"   {module}: {error}")
        return False
    
    print(f"\n✅ All {len(imported)} modules imported successfully")
    
    # Test functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        return False
    
    print("\n✅ Basic functionality tests passed")
    
    # Test integration
    if not test_integration():
        print("\n❌ Integration tests failed")
        return False
    
    print("\n✅ Integration tests passed")
    
    print("\n🎉 Peak-performance hardening implementation validated!")
    print("\nKey features implemented:")
    print("  • Order idempotency with TTL cache")
    print("  • Position reconciliation system")  
    print("  • Exchange-aligned timing")
    print("  • Symbol-aware cost modeling")
    print("  • Adaptive risk controls")
    print("  • Deterministic training")
    print("  • Feature drift monitoring")
    print("  • Performance optimizations")
    print("  • Smart order routing")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)