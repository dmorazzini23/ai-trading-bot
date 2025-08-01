"""Test for the submit_order NameError fix."""
import pytest
import os
from unittest.mock import Mock, patch
import sys

# Set test environment BEFORE any imports
os.environ["PYTEST_RUNNING"] = "1"
os.environ.update({
    "ALPACA_API_KEY": "PKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ALPACA_SECRET_KEY": "SKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD",
    "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
    "WEBHOOK_SECRET": "test-webhook-secret",
    "FLASK_PORT": "9000",
    "BOT_MODE": "balanced",
    "DOLLAR_RISK_LIMIT": "0.02",
    "TESTING": "1",
    "TRADE_LOG_FILE": "test_trades.csv",
    "SEED": "42",
    "RATE_LIMIT_BUDGET": "190",
    "DISABLE_DAILY_RETRAIN": "1",
    "DRY_RUN": "1",
    "SHADOW_MODE": "1",
})


def test_submit_order_with_uninitialized_exec_engine():
    """Test that submit_order raises proper error when _exec_engine is None."""
    # Import after setting environment
    import bot_engine
    from bot_engine import submit_order, BotContext
    
    # Ensure _exec_engine is None
    original_exec_engine = bot_engine._exec_engine
    bot_engine._exec_engine = None
    
    try:
        # Mock market_is_open to return True
        with patch('bot_engine.market_is_open', return_value=True):
            # Create a mock context
            mock_ctx = Mock(spec=BotContext)
            
            # Should raise RuntimeError for uninitialized engine
            with pytest.raises(RuntimeError, match="Execution engine not initialized"):
                submit_order(mock_ctx, "AAPL", 10, "buy")
                
    finally:
        # Restore original state
        bot_engine._exec_engine = original_exec_engine


def test_submit_order_with_market_closed():
    """Test that submit_order returns None when market is closed."""
    # Import after setting environment
    from bot_engine import submit_order, BotContext
    from unittest.mock import Mock
    
    # Mock market_is_open to return False
    with patch('bot_engine.market_is_open', return_value=False):
        mock_ctx = Mock(spec=BotContext)
        result = submit_order(mock_ctx, "AAPL", 10, "buy")
        assert result is None


def test_submit_order_successful_execution():
    """Test that submit_order works correctly when properly initialized."""
    # Import after setting environment
    import bot_engine
    from bot_engine import submit_order, BotContext
    from unittest.mock import Mock
    
    # Mock the execution engine
    mock_exec_engine = Mock()
    mock_order = Mock()
    mock_exec_engine.execute_order.return_value = mock_order
    
    # Set the global _exec_engine
    original_exec_engine = bot_engine._exec_engine
    bot_engine._exec_engine = mock_exec_engine
    
    try:
        # Mock market_is_open to return True
        with patch('bot_engine.market_is_open', return_value=True):
            mock_ctx = Mock(spec=BotContext)
            
            # Should successfully execute order
            result = submit_order(mock_ctx, "AAPL", 10, "buy")
            
            assert result == mock_order
            mock_exec_engine.execute_order.assert_called_once_with("AAPL", 10, "buy")
            
    finally:
        # Restore original state
        bot_engine._exec_engine = original_exec_engine


def test_submit_order_execution_error_propagation():
    """Test that submit_order properly propagates execution errors."""
    # Import after setting environment
    import bot_engine
    from bot_engine import submit_order, BotContext
    from unittest.mock import Mock
    
    # Mock the execution engine to raise an exception
    mock_exec_engine = Mock()
    test_error = Exception("Test execution error")
    mock_exec_engine.execute_order.side_effect = test_error
    
    # Set the global _exec_engine
    original_exec_engine = bot_engine._exec_engine
    bot_engine._exec_engine = mock_exec_engine
    
    try:
        # Mock market_is_open to return True
        with patch('bot_engine.market_is_open', return_value=True):
            mock_ctx = Mock(spec=BotContext)
            
            # Should propagate the execution error
            with pytest.raises(Exception, match="Test execution error"):
                submit_order(mock_ctx, "AAPL", 10, "buy")
                
    finally:
        # Restore original state
        bot_engine._exec_engine = original_exec_engine


if __name__ == "__main__":
    pytest.main([__file__, "-v"])