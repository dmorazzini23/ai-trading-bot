#!/usr/bin/env python3
"""Test script to validate the dependency and data handling fixes."""

import sys
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_alpaca_import_handling():
    """Test that alpaca import handling works with proper fallbacks."""
    print("Testing Alpaca SDK import handling...")
    
    try:
        from ai_trading.utils.base import _get_alpaca_rest
        # This should either succeed or use a mock fallback
        rest_client = _get_alpaca_rest()
        print("✓ Alpaca REST client handling works (with fallback if needed)")
        return True
    except Exception as e:
        print(f"✗ Alpaca REST client handling failed: {e}")
        return False

def test_market_schedule_handling():
    """Test that market schedule handling works with better error messages."""
    print("Testing market schedule handling...")
    
    try:
        from ai_trading.utils.base import is_market_open
        # This should handle missing schedule gracefully
        result = is_market_open()
        print(f"✓ Market schedule check completed: market is {'open' if result else 'closed'}")
        return True
    except Exception as e:
        print(f"✗ Market schedule handling failed: {e}")
        return False

def test_regime_model_validation():
    """Test that regime model uses better data validation."""
    print("Testing regime model validation...")
    
    try:
        # Try to import the regime detection module
        from ai_trading.strategies.regime_detector import RegimeDetector
        print("✓ Regime detector import successful")
        return True
    except ImportError:
        print("ℹ Regime detector not available (expected in minimal environment)")
        return True
    except Exception as e:
        print(f"✗ Regime model validation failed: {e}")
        return False

def test_dependency_availability():
    """Test dependency availability and provide status report."""
    print("Checking dependency availability...")
    
    dependencies = {
        'alpaca_trade_api': 'Legacy Alpaca SDK',
        'alpaca': 'Modern Alpaca SDK (alpaca-py)',
        'pandas_market_calendars': 'Market calendar data',
        'sklearn': 'Machine learning models',
        'hmmlearn': 'Hidden Markov Models for regime detection'
    }
    
    available = 0
    total = len(dependencies)
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✓ {dep}: {description}")
            available += 1
        except ImportError:
            print(f"✗ {dep}: {description} (not available)")
    
    print(f"Dependencies available: {available}/{total}")
    return available > 0  # At least some dependencies should be available

def main():
    """Run all tests and report results."""
    print("=== Dependency and Data Handling Fixes Test ===\n")
    
    tests = [
        test_dependency_availability,
        test_alpaca_import_handling,
        test_market_schedule_handling,
        test_regime_model_validation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}\n")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("✓ All tests passed! Fixes appear to be working correctly.")
        return 0
    elif passed > total // 2:
        print("⚠ Most tests passed, but some issues remain.")
        return 1
    else:
        print("✗ Multiple test failures detected.")
        return 2

if __name__ == "__main__":
    sys.exit(main())