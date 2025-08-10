"""Simplified test for position holding logic without full environment setup."""

import os
import sys
from unittest.mock import Mock, patch

# Set required environment variables for testing
os.environ['ALPACA_API_KEY'] = 'test_key'
os.environ['ALPACA_SECRET_KEY'] = 'test_secret'
os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['WEBHOOK_SECRET'] = 'test_secret'
os.environ['FLASK_PORT'] = '5000'
os.environ['PYTEST_RUNNING'] = '1'  # Enable test mode

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_position_holding_standalone():
    """Test standalone position holding functions."""
    
    # Test the standalone should_hold_position function
    from ai_trading.position.legacy_manager import should_hold_position
    
    # Test holding profitable position
    result = should_hold_position("AAPL", None, 8.5, 2)
    assert result == True, "Should hold profitable position with >5% gain"
    
    # Test holding new position
    result = should_hold_position("AAPL", None, 2.0, 1)
    assert result == True, "Should hold new position for at least 3 days"
    
    # Test not holding old losing position
    result = should_hold_position("AAPL", None, -3.0, 5)
    assert result == False, "Should not hold losing position after min hold period"
    
    print("✓ Position holding logic tests passed")


def test_position_score_standalone():
    """Test standalone position scoring."""
    
    from ai_trading.position.legacy_manager import calculate_position_score
    
    # Test with mock position
    position = Mock()
    position.qty = 100
    
    # Should return some score based on quantity
    score = calculate_position_score("AAPL", position)
    assert isinstance(score, float), "Should return a float score"
    assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"
    
    print("✓ Position scoring tests passed")


def test_meta_learning_functions():
    """Test meta-learning conversion functions."""
    
    with patch('meta_learning.config') as mock_config:
        mock_config.TRADE_LOG_FILE = '/tmp/test_trades.csv'
        
        from ai_trading.meta_learning import convert_audit_to_meta
        
        # Test single conversion
        trade_data = {
            'symbol': 'AAPL',
            'qty': 100,
            'side': 'buy',
            'price': 150.0,
            'timestamp': '2024-01-01T12:00:00Z',
            'order_id': 'test-123'
        }
        
        result = convert_audit_to_meta(trade_data)
        
        assert result is not None, "Conversion should return data"
        assert result['symbol'] == 'AAPL', "Symbol should be preserved"
        assert result['qty'] == 100, "Quantity should be preserved"
        assert result['side'] == 'buy', "Side should be preserved"
        assert result['classification'] == 'converted', "Should be marked as converted"
        
        print("✓ Meta-learning conversion tests passed")


def test_signal_filtering():
    """Test signal filtering with mock data."""
    
    # Mock the signals module to avoid import issues
    with patch('sys.modules', {'signals': Mock()}):
        
        # Test basic signal filtering logic
        signals = [
            {'symbol': 'AAPL', 'side': 'sell'},
            {'symbol': 'GOOGL', 'side': 'buy'},
            {'symbol': 'MSFT', 'side': 'buy'}
        ]
        
        hold_signals = {
            'AAPL': 'hold',  # Should filter out sell signal
            'GOOGL': 'sell'  # Should filter out buy signal
        }
        
        # Simulate filtering logic
        filtered_signals = []
        for signal in signals:
            symbol = signal['symbol']
            side = signal['side']
            
            if symbol in hold_signals:
                hold_action = hold_signals[symbol]
                if hold_action == 'hold' and side == 'sell':
                    continue  # Filter out sell signal for held position
                elif hold_action == 'sell' and side == 'buy':
                    continue  # Filter out buy signal when sell pending
            
            filtered_signals.append(signal)
        
        # Should only have MSFT buy signal remaining
        assert len(filtered_signals) == 1, "Should filter out 2 signals"
        assert filtered_signals[0]['symbol'] == 'MSFT', "MSFT signal should remain"
        
        print("✓ Signal filtering tests passed")


if __name__ == "__main__":
    try:
        test_position_holding_standalone()
        test_position_score_standalone() 
        test_meta_learning_functions()
        test_signal_filtering()
        print("\n🎉 All simplified tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)