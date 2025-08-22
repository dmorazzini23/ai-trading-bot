"""Test position holding logic and meta-learning triggers."""

import os
import sys
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def test_position_manager_should_hold_profit():
    """Test that profitable positions are held."""
    from ai_trading.position.legacy_manager import PositionManager

    # Create mock context
    ctx = Mock()
    ctx.data_fetcher = Mock()
    ctx.api = Mock()

    # Mock position data
    position = Mock()
    position.symbol = "AAPL"
    position.qty = 100
    position.avg_entry_price = 150.0

    # Mock data fetcher to return empty data (for testing fallback)
    ctx.data_fetcher.get_minute_df.return_value = Mock()
    ctx.data_fetcher.get_minute_df.return_value.empty = True
    ctx.data_fetcher.get_daily_df.return_value = Mock()
    ctx.data_fetcher.get_daily_df.return_value.empty = True

    # Create position manager
    pm = PositionManager(ctx)

    # Test holding profitable position (>5% gain)
    result = pm.should_hold_position("AAPL", position, 8.5, 2)
    assert result is True, "Should hold profitable position with >5% gain"

    # Test holding new position (<3 days)
    result = pm.should_hold_position("AAPL", position, 2.0, 1)
    assert result is True, "Should hold new position for at least 3 days"

    # Test not holding losing position
    result = pm.should_hold_position("AAPL", position, -3.0, 5)
    assert result is False, "Should not hold losing position after min hold period"


def test_position_hold_signals_generation():
    """Test position hold signal generation."""
    from ai_trading.signals import generate_position_hold_signals

    # Create mock context with position manager
    ctx = Mock()
    ctx.position_manager = Mock()

    # Mock current positions
    positions = [
        Mock(symbol="AAPL", qty=100, avg_entry_price=150.0),
        Mock(symbol="GOOGL", qty=50, avg_entry_price=2800.0),
    ]

    # Mock hold signals from position manager
    ctx.position_manager.get_hold_signals.return_value = {
        "AAPL": "hold",
        "GOOGL": "sell"
    }

    # Test signal generation
    signals = generate_position_hold_signals(ctx, positions)

    assert signals == {"AAPL": "hold", "GOOGL": "sell"}
    ctx.position_manager.get_hold_signals.assert_called_once_with(positions)


def test_signals_enhancement_with_position_logic():
    """Test that signals are enhanced with position holding logic."""
    from ai_trading.signals import enhance_signals_with_position_logic

    # Create mock signals
    signal1 = Mock()
    signal1.symbol = "AAPL"
    signal1.side = "sell"

    signal2 = Mock()
    signal2.symbol = "GOOGL"
    signal2.side = "buy"

    signals = [signal1, signal2]

    # Mock context
    ctx = Mock()

    # Mock hold signals - AAPL should be held, GOOGL can be sold/bought
    hold_signals = {
        "AAPL": "hold",
        "GOOGL": "sell"
    }

    # Test enhancement
    enhanced = enhance_signals_with_position_logic(signals, ctx, hold_signals)

    # AAPL sell signal should be filtered out (converted to hold)
    # GOOGL buy signal should be filtered out (sell pending)
    assert len(enhanced) == 0, "Both signals should be filtered out"


def test_meta_learning_trigger():
    """Test meta-learning conversion trigger."""
    with patch('meta_learning.config') as mock_config, \
         patch('meta_learning.pd') as mock_pd, \
         patch('ai_trading.meta_learning.Path') as mock_path:

        from ai_trading.meta_learning import trigger_meta_learning_conversion

        # Mock config
        mock_config.TRADE_LOG_FILE = '/tmp/test_trades.csv'

        # Mock file exists
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock quality report
        with patch('meta_learning.validate_trade_data_quality') as mock_validate:
            mock_validate.return_value = {
                'file_exists': True,
                'row_count': 5,
                'mixed_format_detected': True
            }

            # Mock DataFrame and conversion
            mock_df = Mock()
            mock_pd.read_csv.return_value = mock_df

            with patch('meta_learning._convert_audit_to_meta_format') as mock_convert:
                mock_converted_df = Mock()
                mock_converted_df.empty = False
                mock_convert.return_value = mock_converted_df

                # Test trade data
                trade_data = {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'side': 'buy',
                    'price': 150.0,
                    'timestamp': '2024-01-01T12:00:00Z',
                    'order_id': 'test-123',
                    'status': 'filled'
                }

                # Test trigger
                result = trigger_meta_learning_conversion(trade_data)

                assert result is True, "Meta-learning conversion should succeed"
                mock_validate.assert_called_once()
                mock_convert.assert_called_once()


def test_position_manager_cleanup():
    """Test position manager cleanup of stale positions."""
    from ai_trading.position.legacy_manager import PositionManager

    # Create mock context
    ctx = Mock()
    ctx.api = Mock()

    # Mock current positions (only AAPL exists now)
    current_positions = [Mock(symbol="AAPL")]
    ctx.api.get_all_positions.return_value = current_positions

    # Create position manager with existing tracked positions
    pm = PositionManager(ctx)
    pm.positions = {
        "AAPL": Mock(),
        "GOOGL": Mock(),  # This should be cleaned up
        "MSFT": Mock()    # This should be cleaned up
    }

    # Test cleanup
    pm.cleanup_stale_positions()

    # Only AAPL should remain
    assert "AAPL" in pm.positions
    assert "GOOGL" not in pm.positions
    assert "MSFT" not in pm.positions


def test_position_score_calculation():
    """Test position scoring calculation."""
    from ai_trading.position.legacy_manager import PositionManager

    # Create mock context
    ctx = Mock()
    ctx.data_fetcher = Mock()

    # Mock empty data for fallback testing
    empty_df = Mock()
    empty_df.empty = True
    ctx.data_fetcher.get_minute_df.return_value = empty_df
    ctx.data_fetcher.get_daily_df.return_value = empty_df

    # Create position manager
    pm = PositionManager(ctx)

    # Mock position data
    position = Mock()
    position.avg_entry_price = 100.0
    position.qty = 100

    # Test with empty data (should return 0.0)
    score = pm.calculate_position_score("AAPL", position)
    assert score == 0.0, "Should return 0 for position with no price data"


if __name__ == "__main__":
    # Run basic tests
    test_position_manager_should_hold_profit()
    test_position_hold_signals_generation()
    test_signals_enhancement_with_position_logic()
    test_meta_learning_trigger()
    test_position_manager_cleanup()
    test_position_score_calculation()
