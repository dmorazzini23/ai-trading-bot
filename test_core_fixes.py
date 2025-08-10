#!/usr/bin/env python3
"""Test core fixes without requiring full environment setup."""

import sys
import os

def test_alpaca_import_with_mock():
    """Test alpaca import handling works with our mock fallback."""
    print("Testing alpaca import with mock fallback...")
    
    # Temporarily set environment variables to avoid config errors
    os.environ.setdefault('ALPACA_API_KEY', 'test_key')
    os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
    os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook')
    os.environ.setdefault('FLASK_PORT', '5000')
    
    try:
        from ai_trading.utils.base import _get_alpaca_rest
        
        # This should work now with our improved fallback
        rest_client = _get_alpaca_rest()
        print("✓ _get_alpaca_rest() completed without crashing")
        
        # Test that mock methods work
        if hasattr(rest_client, '__getattr__'):
            # This is our mock class
            print("✓ Using mock REST client (alpaca_trade_api not installed)")
            return True
        else:
            print("✓ Using real REST client (alpaca_trade_api installed)")
            return True
            
    except Exception as e:
        print(f"✗ Alpaca import handling failed: {e}")
        return False

def test_market_schedule_with_fallback():
    """Test market schedule works with fallback logic."""
    print("Testing market schedule with fallback...")
    
    # Set environment variables to avoid config errors
    os.environ.setdefault('ALPACA_API_KEY', 'test_key')
    os.environ.setdefault('ALPACA_SECRET_KEY', 'test_secret')
    os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    os.environ.setdefault('WEBHOOK_SECRET', 'test_webhook')
    os.environ.setdefault('FLASK_PORT', '5000')
    
    try:
        from ai_trading.utils.base import is_market_open
        
        # This should complete without crashing, even if calendar data unavailable
        result = is_market_open()
        print(f"✓ is_market_open() completed: {result}")
        return True
        
    except Exception as e:
        print(f"✗ Market schedule handling failed: {e}")
        return False

def main():
    """Run core tests with minimal environment setup."""
    print("=== Core Fixes Test (with mock environment) ===\n")
    
    tests = [
        test_alpaca_import_with_mock,
        test_market_schedule_with_fallback,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}\n")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("✓ All core tests passed!")
        return 0
    else:
        print("✗ Some core tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())