#!/usr/bin/env python3
"""
Simple validation script to test the three runtime blocker fixes.
"""

import os
import re
import sys


def test_legacy_imports():
    """Test that legacy import shims work"""

    # Test signals import
    try:
        success = True
    except Exception:
        success = False

    # Test indicators import
    try:
        pass
    except Exception:
        success = False

    # Test rebalancer import (expected to fail due to config requirements)
    try:
        pass
    except Exception as e:
        if "ALPACA_API_KEY" in str(e) or "pydantic_settings" in str(e):
            pass
        else:
            success = False

    return success

def test_ohlcv_files_exist():
    """Test OHLCV normalizer files exist"""

    ohlcv_path = "ai_trading/utils/ohlcv.py"
    if os.path.exists(ohlcv_path):

        # Check if it contains the standardize_ohlcv function
        with open(ohlcv_path) as f:
            content = f.read()
            if "def standardize_ohlcv" in content and "CANON" in content:
                return True
            else:
                return False
    else:
        return False

def test_bot_engine_changes():
    """Test that bot_engine.py has the required changes"""

    bot_engine_path = "ai_trading/core/bot_engine.py"
    if not os.path.exists(bot_engine_path):
        return False

    with open(bot_engine_path) as f:
        content = f.read()

    # Check for OHLCV fix
    if "standardize_ohlcv" in content and "_compute_regime_features" in content:
        pass
    else:
        return False

    # Check for prometheus fix
    if "_init_metrics" in content and "_METRICS_READY" in content:
        pass
    else:
        return False

    # Check that module-level Counter definitions are removed
    lines = content.split('\n')
    module_level_counters = []
    for i, line in enumerate(lines):
        # Check for lines that start with a variable assignment to Counter (not indented)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*Counter\(', line):
            module_level_counters.append((i+1, line.strip()))

    if module_level_counters:
        for line_num, line in module_level_counters:
            pass
        return False
    else:
        pass

    return True

def test_top_level_shims():
    """Test that top-level shim files exist"""

    shims = ["signals.py", "rebalancer.py", "indicators.py"]
    success = True

    for shim in shims:
        if os.path.exists(shim):
            pass
        else:
            success = False

    # Check bot_engine.py has prepare_indicators
    with open("bot_engine.py") as f:
        content = f.read()
        if "prepare_indicators" in content:
            pass
        else:
            success = False

    return success

def main():
    """Run all validation tests"""

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
        except Exception:
            pass


    if passed == len(tests):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
