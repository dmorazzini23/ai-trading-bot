#!/usr/bin/env python3
"""Test script to validate Alpaca null safety fixes."""

import os
import sys
import types
import unittest.mock
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_safe_alpaca_get_account_with_none_api():
    """Test safe_alpaca_get_account handles None api gracefully."""
    print("Testing safe_alpaca_get_account with None api...")
    
    # Mock the bot_engine module dependencies
    with patch('bot_engine.check_alpaca_available', return_value=False):
        with patch('bot_engine.logger') as mock_logger:
            import bot_engine
            
            # Create a mock context with None api
            ctx = types.SimpleNamespace(api=None)
            
            # Call the function
            result = bot_engine.safe_alpaca_get_account(ctx)
            
            # Verify it returns None
            assert result is None, f"Expected None, got {result}"
            
            # Verify warning was logged
            mock_logger.warning.assert_called_with("Alpaca trading client unavailable for account fetch - skipping")
            
            print("✓ safe_alpaca_get_account handles None api correctly")

def test_check_pdt_rule_with_unavailable_alpaca():
    """Test check_pdt_rule handles unavailable Alpaca gracefully."""
    print("Testing check_pdt_rule with unavailable Alpaca...")
    
    with patch('bot_engine.safe_alpaca_get_account', return_value=None):
        with patch('bot_engine.logger') as mock_logger:
            import bot_engine
            
            # Create a mock context
            ctx = types.SimpleNamespace(api=None)
            
            # Call the function
            result = bot_engine.check_pdt_rule(ctx)
            
            # Verify it returns False (no PDT blocking)
            assert result is False, f"Expected False, got {result}"
            
            # Verify info message was logged
            mock_logger.info.assert_called_with("PDT_CHECK_SKIPPED - Alpaca unavailable, assuming no PDT restrictions")
            
            print("✓ check_pdt_rule handles unavailable Alpaca correctly")

def test_cancel_all_open_orders_with_none_api():
    """Test cancel_all_open_orders handles None api gracefully."""
    print("Testing cancel_all_open_orders with None api...")
    
    with patch('bot_engine.check_alpaca_available', return_value=False):
        with patch('bot_engine.logger') as mock_logger:
            import bot_engine
            
            # Create a mock context with None api
            ctx = types.SimpleNamespace(api=None)
            
            # Call the function - should not raise an exception
            try:
                bot_engine.cancel_all_open_orders(ctx)
                print("✓ cancel_all_open_orders handles None api without exception")
            except Exception as e:
                print(f"✗ cancel_all_open_orders raised exception: {e}")
                sys.exit(1)
            
            # Verify info message was logged
            mock_logger.info.assert_called_with("Skipping cancel_all_open_orders - Alpaca unavailable")

def test_alpaca_unavailable_flag():
    """Test that ALPACA_AVAILABLE flag is properly set when imports fail."""
    print("Testing ALPACA_AVAILABLE flag...")
    
    # Import bot_engine to check the flag
    import bot_engine
    
    # The flag should be available
    assert hasattr(bot_engine, 'ALPACA_AVAILABLE'), "ALPACA_AVAILABLE flag not found"
    
    print(f"✓ ALPACA_AVAILABLE flag is set to: {bot_engine.ALPACA_AVAILABLE}")

def test_check_alpaca_available_function():
    """Test the check_alpaca_available utility function."""
    print("Testing check_alpaca_available function...")
    
    import bot_engine
    
    # Test with mock values
    with patch('bot_engine.ALPACA_AVAILABLE', False):
        with patch('bot_engine.logger') as mock_logger:
            result = bot_engine.check_alpaca_available("test operation")
            
            assert result is False, f"Expected False, got {result}"
            mock_logger.warning.assert_called_with("Alpaca trading client unavailable for test operation - skipping")
            
    print("✓ check_alpaca_available function works correctly")

if __name__ == "__main__":
    print("Running Alpaca null safety tests...\n")
    
    try:
        test_alpaca_unavailable_flag()
        test_check_alpaca_available_function()
        test_safe_alpaca_get_account_with_none_api()
        test_check_pdt_rule_with_unavailable_alpaca()
        test_cancel_all_open_orders_with_none_api()
        
        print("\n✅ All tests passed! Alpaca null safety fixes are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)