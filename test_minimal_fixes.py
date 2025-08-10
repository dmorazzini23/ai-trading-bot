#!/usr/bin/env python3
"""Minimal test to validate import error handling without requiring environment setup."""

import sys
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def test_alpaca_trade_api_handling():
    """Test that code handles missing alpaca_trade_api gracefully."""
    print("Testing alpaca_trade_api import handling...")
    
    try:
        # Test direct import attempt
        try:
            import alpaca_trade_api
            print("✓ alpaca_trade_api is available")
            return True
        except ImportError:
            print("ℹ alpaca_trade_api not installed (expected without pip install)")
            
            # Test that our fallback handling works
            try:
                from ai_trading.utils.base import _get_alpaca_rest
                print("ℹ Fallback import mechanism available") 
                return True
            except Exception as e:
                print(f"✗ Fallback mechanism failed: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_requirements_files():
    """Test that alpaca_trade_api has been added to requirements."""
    print("Testing requirements file updates...")
    
    files_to_check = ['requirements.txt', 'pyproject.toml']
    found_in_files = []
    
    for filename in files_to_check:
        try:
            with open(filename, 'r') as f:
                content = f.read()
                if 'alpaca-trade-api' in content:
                    found_in_files.append(filename)
                    print(f"✓ alpaca-trade-api found in {filename}")
        except FileNotFoundError:
            print(f"⚠ {filename} not found")
    
    if found_in_files:
        print(f"✓ alpaca-trade-api dependency added to: {', '.join(found_in_files)}")
        return True
    else:
        print("✗ alpaca-trade-api not found in any requirements file")
        return False

def test_import_fallbacks():
    """Test that modules can be imported even without dependencies."""
    print("Testing import fallback mechanisms...")
    
    modules_to_test = [
        'ai_trading.utils.base',
        'ai_trading.strategies.regime_detector',
    ]
    
    success_count = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module} imports successfully")
            success_count += 1
        except Exception as e:
            print(f"✗ {module} import failed: {e}")
    
    print(f"Import success rate: {success_count}/{len(modules_to_test)}")
    return success_count > 0

def main():
    """Run minimal tests that don't require environment setup."""
    print("=== Minimal Dependency Fixes Test ===\n")
    
    tests = [
        test_requirements_files,
        test_alpaca_trade_api_handling,
        test_import_fallbacks,
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
        print("✓ All tests passed!")
        return 0
    elif passed > 0:
        print("⚠ Some tests passed, fixes are partially working.")
        return 1
    else:
        print("✗ All tests failed.")
        return 2

if __name__ == "__main__":
    sys.exit(main())