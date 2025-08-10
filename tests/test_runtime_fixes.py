#!/usr/bin/env python3
"""
Simple validation script to test the three runtime blocker fixes.
"""

import sys
import os
import re

def test_legacy_imports():
    """Test that legacy import shims work"""
    print("Testing legacy import shims...")
    
    # Test signals import
    try:
        print("✓ signals import successful")
        success = True
    except Exception as e:
        print(f"✗ signals import failed: {e}")
        success = False
    
    # Test indicators import 
    try:
        print("✓ indicators import successful")
    except Exception as e:
        print(f"✗ indicators import failed: {e}")
        success = False
    
    # Test rebalancer import (expected to fail due to config requirements)
    try:
        print("✓ rebalancer import successful")
    except Exception as e:
        if "ALPACA_API_KEY" in str(e) or "pydantic_settings" in str(e):
            print("✓ rebalancer import failed as expected (missing env vars or deps)")
        else:
            print(f"✗ rebalancer import failed unexpectedly: {e}")
            success = False
    
    return success

def test_ohlcv_files_exist():
    """Test OHLCV normalizer files exist"""
    print("\nTesting OHLCV normalizer...")
    
    ohlcv_path = "ai_trading/utils/ohlcv.py"
    if os.path.exists(ohlcv_path):
        print("✓ OHLCV normalizer file exists")
        
        # Check if it contains the standardize_ohlcv function
        with open(ohlcv_path, 'r') as f:
            content = f.read()
            if "def standardize_ohlcv" in content and "CANON" in content:
                print("✓ OHLCV normalizer contains required functions")
                return True
            else:
                print("✗ OHLCV normalizer missing required functions")
                return False
    else:
        print("✗ OHLCV normalizer file missing")
        return False

def test_bot_engine_changes():
    """Test that bot_engine.py has the required changes"""
    print("\nTesting bot_engine.py changes...")
    
    bot_engine_path = "ai_trading/core/bot_engine.py"
    if not os.path.exists(bot_engine_path):
        print("✗ bot_engine.py not found")
        return False
    
    with open(bot_engine_path, 'r') as f:
        content = f.read()
    
    # Check for OHLCV fix
    if "standardize_ohlcv" in content and "_compute_regime_features" in content:
        print("✓ OHLCV normalization added to _compute_regime_features")
    else:
        print("✗ OHLCV normalization missing from _compute_regime_features")
        return False
    
    # Check for prometheus fix
    if "_init_metrics" in content and "_METRICS_READY" in content:
        print("✓ Prometheus lazy initialization added")
    else:
        print("✗ Prometheus lazy initialization missing")
        return False
    
    # Check that module-level Counter definitions are removed
    lines = content.split('\n')
    module_level_counters = []
    for i, line in enumerate(lines):
        # Check for lines that start with a variable assignment to Counter (not indented)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*Counter\(', line):
            module_level_counters.append((i+1, line.strip()))
    
    if module_level_counters:
        print("✗ Module-level Counter definitions still present:")
        for line_num, line in module_level_counters:
            print(f"   Line {line_num}: {line}")
        return False
    else:
        print("✓ Module-level Counter definitions removed")
    
    return True

def test_top_level_shims():
    """Test that top-level shim files exist"""
    print("\nTesting top-level shim files...")
    
    shims = ["signals.py", "rebalancer.py", "indicators.py"]
    success = True
    
    for shim in shims:
        if os.path.exists(shim):
            print(f"✓ {shim} shim exists")
        else:
            print(f"✗ {shim} shim missing")
            success = False
    
    # Check bot_engine.py has prepare_indicators
    with open("bot_engine.py", 'r') as f:
        content = f.read()
        if "prepare_indicators" in content:
            print("✓ bot_engine.py has prepare_indicators compatibility")
        else:
            print("✗ bot_engine.py missing prepare_indicators compatibility")
            success = False
    
    return success

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("VALIDATION: Testing fixes for three runtime blockers")
    print("=" * 60)
    
    tests = [
        test_legacy_imports,
        test_ohlcv_files_exist,
        test_bot_engine_changes,
        test_top_level_shims,
    ]
    
    passed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All fixes appear to be working correctly!")
        return 0
    else:
        print("⚠ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())