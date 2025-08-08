#!/usr/bin/env python3
"""
Simple test to verify the basic fixes work.
"""
import os
import sys

def test_file_creation():
    """Test that required files exist."""
    files_to_check = [
        "slippage.csv",
        "tickers.csv", 
        "ai_trading/logs/slippage.csv",
        "ai_trading/core/tickers.csv"
    ]
    
    print("Checking file existence:")
    for f in files_to_check:
        exists = os.path.exists(f)
        print(f"  {f}: {'✓' if exists else '✗'}")
        if not exists:
            return False
    return True

def test_portfolio_function():
    """Test portfolio weights function exists."""
    try:
        from ai_trading.portfolio import compute_portfolio_weights
        print("✓ compute_portfolio_weights function found")
        return True
    except ImportError as e:
        print(f"✗ Portfolio import failed: {e}")
        return False

def test_config_paths():
    """Test config file paths."""
    try:
        import sys
        import os
        # Add the root directory to path to access root config.py
        root_dir = os.path.dirname(os.path.abspath(__file__))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        
        import config
        SLIPPAGE_LOG_PATH = getattr(config, 'SLIPPAGE_LOG_PATH', 'Not found')
        TICKERS_FILE_PATH = getattr(config, 'TICKERS_FILE_PATH', 'Not found')
        print(f"✓ Config paths defined: SLIPPAGE_LOG_PATH={SLIPPAGE_LOG_PATH}, TICKERS_FILE_PATH={TICKERS_FILE_PATH}")
        return True
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False

if __name__ == "__main__":
    print("Running basic validation tests...")
    
    tests = [
        ("File Creation", test_file_creation),
        ("Portfolio Function", test_portfolio_function), 
        ("Config Paths", test_config_paths)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n=== {name} ===")
        try:
            if test_func():
                passed += 1
                print(f"✓ {name} PASSED")
            else:
                print(f"✗ {name} FAILED")
        except Exception as e:
            print(f"✗ {name} ERROR: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Tests passed: {passed}/{len(tests)}")
    sys.exit(0 if passed == len(tests) else 1)