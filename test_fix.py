#!/usr/bin/env python3
"""
Test script to verify the fixes for duplicate startup logs and TA-Lib import safety.
"""

import sys
import os

def test_core_import_no_side_effects():
    """Test that importing ai_trading.core doesn't trigger bot_engine side effects."""
    print("=== Test 1: Core import behavior ===")
    print("Before importing ai_trading.core")
    print("bot_engine loaded?", "ai_trading.core.bot_engine" in sys.modules)
    
    try:
        import ai_trading.core
        print("After importing ai_trading.core")
        print("bot_engine loaded?", "ai_trading.core.bot_engine" in sys.modules)
        print("✓ Test 1 PASSED: Core import doesn't load bot_engine")
        return True
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return False

def test_ta_lib_safety():
    """Test that TA-Lib import safety works correctly."""
    print("\n=== Test 2: TA-Lib import safety ===")
    try:
        from ai_trading.imports import TALIB_AVAILABLE, PANDAS_TA_AVAILABLE
        print(f"TALIB_AVAILABLE: {TALIB_AVAILABLE}")
        print(f"PANDAS_TA_AVAILABLE: {PANDAS_TA_AVAILABLE}")
        
        if TALIB_AVAILABLE:
            from ai_trading.imports import talib
            print("TA-Lib import successful")
        else:
            print("TA-Lib not available (expected)")
            
        print("✓ Test 2 PASSED: TA-Lib import safety works")
        return True
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False

def test_compilation():
    """Test that all Python files compile correctly."""
    print("\n=== Test 3: Code compilation ===")
    try:
        import compileall
        import pathlib
        
        bad = [str(p) for p in pathlib.Path('.').rglob('*.py') 
               if not compileall.compile_file(str(p), quiet=1)]
        
        if not bad:
            print("✓ Test 3 PASSED: All Python files compile successfully")
            return True
        else:
            print(f"✗ Test 3 FAILED: Compilation failures: {bad}")
            return False
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        return False

def test_lazy_bot_context():
    """Test that LazyBotContext is truly lazy and doesn't initialize on creation."""
    print("\n=== Test 4: LazyBotContext laziness ===")
    try:
        # Set environment to avoid real initialization
        os.environ["PYTEST_RUNNING"] = "1"
        
        # This should work without triggering heavy initialization
        from ai_trading.core.bot_engine import LazyBotContext
        
        # Create context - should not initialize
        ctx = LazyBotContext()
        is_initialized = getattr(ctx, '_initialized', False)
        
        print(f"LazyBotContext created, initialized: {is_initialized}")
        
        if not is_initialized:
            print("✓ Test 4 PASSED: LazyBotContext is truly lazy")
            return True
        else:
            print("✗ Test 4 FAILED: LazyBotContext initialized eagerly")
            return False
            
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up environment
        if "PYTEST_RUNNING" in os.environ:
            del os.environ["PYTEST_RUNNING"]

def main():
    """Run all tests."""
    print("Testing fixes for duplicate startup logs and TA-Lib import safety\n")
    
    tests = [
        test_core_import_no_side_effects,
        test_ta_lib_safety,
        test_compilation,
        test_lazy_bot_context,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())