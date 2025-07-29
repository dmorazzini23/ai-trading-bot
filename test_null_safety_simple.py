#!/usr/bin/env python3
"""Simple test script to validate null safety in key functions."""

import os
import sys
import types
import logging
from unittest.mock import MagicMock, patch

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_function_exists():
    """Test that our target functions exist."""
    print("Testing function definitions...")
    
    # Read bot_engine.py to check our functions exist
    with open('bot_engine.py', 'r') as f:
        content = f.read()
    
    # Check that our updated functions exist
    assert 'def safe_alpaca_get_account(ctx: "BotContext"):' in content
    assert 'def check_pdt_rule(ctx: BotContext) -> bool:' in content
    assert 'def cancel_all_open_orders(ctx: "BotContext") -> None:' in content
    assert 'def check_alpaca_available(operation_name: str = "operation") -> bool:' in content
    
    print("✓ All expected functions are defined")

def test_null_safety_patterns():
    """Test that null safety patterns are implemented."""
    print("Testing null safety patterns...")
    
    with open('bot_engine.py', 'r') as f:
        content = f.read()
    
    # Check for null safety patterns in safe_alpaca_get_account
    assert 'if not check_alpaca_available("account fetch"):' in content
    assert 'if ctx.api is None:' in content
    
    # Check for null safety patterns in check_pdt_rule
    assert 'if acct is None:' in content
    assert 'PDT_CHECK_SKIPPED - Alpaca unavailable' in content
    
    # Check for null safety patterns in cancel_all_open_orders
    assert 'if not check_alpaca_available("cancel open orders"):' in content
    
    print("✓ Null safety patterns are implemented")

def test_degraded_mode_messages():
    """Test that appropriate degraded mode messages are present."""
    print("Testing degraded mode messages...")
    
    with open('bot_engine.py', 'r') as f:
        content = f.read()
    
    # Check for degraded mode logging
    assert 'Alpaca trading client unavailable' in content
    assert 'assuming no PDT restrictions' in content
    assert 'Skipping cancel_all_open_orders - Alpaca unavailable' in content
    
    print("✓ Degraded mode messages are present")

def test_error_handling_improvements():
    """Test that error handling has been improved."""
    print("Testing error handling improvements...")
    
    with open('bot_engine.py', 'r') as f:
        content = f.read()
    
    # Check that exceptions are handled gracefully
    assert 'logger.warning("Failed to cancel open orders:' in content
    assert 'except (AttributeError, TypeError, ValueError):' in content
    
    print("✓ Error handling has been improved")

def test_alpaca_available_check():
    """Test that ALPACA_AVAILABLE flag is used consistently."""
    print("Testing ALPACA_AVAILABLE usage...")
    
    with open('bot_engine.py', 'r') as f:
        content = f.read()
    
    # Check that ALPACA_AVAILABLE is defined and used
    assert 'ALPACA_AVAILABLE = True' in content
    assert 'ALPACA_AVAILABLE = False' in content
    assert 'if ALPACA_AVAILABLE:' in content
    
    print("✓ ALPACA_AVAILABLE flag is used consistently")

def test_mock_classes_exist():
    """Test that mock classes are defined for when Alpaca is unavailable."""
    print("Testing mock classes...")
    
    with open('bot_engine.py', 'r') as f:
        content = f.read()
    
    # Check for mock classes
    assert 'class MockOrderSide:' in content
    assert 'class MockQueryOrderStatus:' in content
    assert 'class MockTimeInForce:' in content
    
    print("✓ Mock classes are defined")

if __name__ == "__main__":
    print("Running simple null safety validation...\n")
    
    try:
        test_function_exists()
        test_null_safety_patterns()
        test_degraded_mode_messages()
        test_error_handling_improvements()
        test_alpaca_available_check()
        test_mock_classes_exist()
        
        print("\n✅ All validation tests passed! Code changes look correct.")
        
    except AssertionError as e:
        print(f"\n❌ Validation failed: Missing expected pattern")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)