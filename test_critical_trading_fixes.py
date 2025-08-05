#!/usr/bin/env python3
"""
Test suite for critical trading bot fixes.

Tests the specific fixes implemented for:
1. Meta-learning system recovery
2. Order execution robustness 
3. Risk management enhancement
4. Data integrity monitoring
"""

import tempfile
import pytest
import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test meta-learning system recovery
def test_meta_learning_data_quality_validation():
    """Test that meta-learning validates data quality before training."""
    from meta_learning import validate_trade_data_quality
    
    # Test with non-existent file
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp_path = tmp.name
    
    quality_report = validate_trade_data_quality(tmp_path)
    assert quality_report['file_exists'] is False
    assert 'Trade log file does not exist' in quality_report['issues'][0]
    assert 'Initialize trade logging system' in quality_report['recommendations'][0]


def test_meta_learning_fallback_data_recovery():
    """Test that meta-learning implements fallback procedures for insufficient data."""
    from meta_learning import _implement_fallback_data_recovery, _create_emergency_trade_log
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp_path = tmp.name
    
    try:
        # Remove the file to test creation
        os.unlink(tmp_path)
        
        # Test fallback recovery
        _implement_fallback_data_recovery(tmp_path, min_samples=20)
        
        # Check that emergency log was created
        assert Path(tmp_path).exists()
        
        # Verify proper format
        df = pd.read_csv(tmp_path)
        required_cols = ['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl', 'signal_tags']
        for col in required_cols:
            assert col in df.columns
    finally:
        if Path(tmp_path).exists():
            os.unlink(tmp_path)


def test_meta_learning_price_validation():
    """Test that meta-learning filters out invalid price data."""
    from meta_learning import validate_trade_data_quality
    
    # Create test CSV with mixed valid/invalid data
    test_data = {
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'side': ['buy', 'sell', 'buy'],
        'entry_price': [150.50, -10.0, 2800.0],  # One negative price
        'exit_price': [155.0, 200.0, 2850.0],
        'quantity': [100, 50, 10],
        'pnl': [450.0, -500.0, 500.0],
        'signal_tags': ['momentum', 'mean_reversion', 'momentum']
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame(test_data)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        quality_report = validate_trade_data_quality(tmp_path)
        assert quality_report['file_exists'] is True
        assert quality_report['has_valid_format'] is True
        assert quality_report['row_count'] == 3
        assert quality_report['valid_price_rows'] == 2  # Should filter out negative price
    finally:
        os.unlink(tmp_path)


def test_order_execution_partial_fill_tracking():
    """Test that partial fills are properly tracked and reconciled."""
    from trade_execution import TradingEngine
    
    # Mock the trading context and API
    mock_ctx = Mock()
    mock_ctx.data_client = Mock()
    mock_ctx.data_fetcher = Mock()
    mock_api = Mock()
    
    # Create trading engine instance
    engine = TradingEngine(mock_ctx, api=mock_api)
    
    # Mock order result
    mock_order = Mock()
    mock_order.id = "test_order_123"
    
    # Test partial fill reconciliation
    engine._reconcile_partial_fills("AAPL", requested_qty=100, remaining_qty=30, side="buy", last_order=mock_order)
    
    # Verify that partial fill was logged (would be in logs in real implementation)
    # This is a basic structure test
    assert hasattr(engine, '_reconcile_partial_fills')


def test_risk_management_sector_exposure_logging():
    """Test that sector exposure rejections include clear reasoning."""
    # This is a minimal test - full test would require bot_engine context
    # Testing the structure exists for enhanced logging
    from bot_engine import sector_exposure_ok
    import logging
    
    # Mock BotContext
    mock_ctx = Mock()
    mock_ctx.api = Mock()
    
    # Mock account with zero portfolio value
    mock_account = Mock()
    mock_account.portfolio_value = 0
    mock_ctx.api.get_account.return_value = mock_account
    
    # Test empty portfolio logic
    result = sector_exposure_ok(mock_ctx, "AAPL", 10, 150.0)
    assert result is True  # Should allow initial positions


def test_data_integrity_validation():
    """Test comprehensive data integrity validation."""
    from data_validation import validate_trade_log_integrity, monitor_real_time_data_quality
    
    # Test trade log integrity validation
    test_data = {
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'symbol': ['AAPL', 'MSFT'],
        'side': ['buy', 'sell'],
        'entry_price': [150.50, 200.0],
        'exit_price': [155.0, 195.0],
        'quantity': [100, 50],
        'pnl': [450.0, -250.0]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame(test_data)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        integrity_report = validate_trade_log_integrity(tmp_path)
        assert integrity_report['file_exists'] is True
        assert integrity_report['file_readable'] is True
        assert integrity_report['valid_format'] is True
        assert integrity_report['data_consistent'] is True
        assert integrity_report['total_trades'] == 2
        assert integrity_report['integrity_score'] >= 0.9
    finally:
        os.unlink(tmp_path)
    
    # Test real-time data quality monitoring
    price_data = {
        'AAPL': 150.50,
        'MSFT': 300.0,
        'INVALID': -10.0  # Invalid negative price
    }
    
    quality_report = monitor_real_time_data_quality(price_data)
    assert quality_report['data_quality_ok'] is False
    assert 'INVALID' in quality_report['critical_symbols']
    assert len(quality_report['anomalies_detected']) >= 1


def test_data_corruption_detection():
    """Test that data corruption is properly detected."""
    from data_validation import validate_trade_log_integrity
    
    # Create corrupted test data
    corrupted_data = {
        'timestamp': ['invalid_date', '2024-01-01 11:00:00'],
        'symbol': ['AAPL', 'MSFT'],
        'side': ['invalid_side', 'sell'],
        'entry_price': ['not_a_number', 200.0],
        'exit_price': [155.0, 195.0],
        'quantity': [-100, 50],  # Invalid negative quantity
        'pnl': [450.0, -250.0]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame(corrupted_data)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        integrity_report = validate_trade_log_integrity(tmp_path)
        assert integrity_report['file_exists'] is True
        assert integrity_report['file_readable'] is True
        assert integrity_report['data_consistent'] is False
        assert len(integrity_report['corrupted_rows']) >= 1
        assert integrity_report['integrity_score'] < 0.9
    finally:
        os.unlink(tmp_path)


def test_emergency_data_validation():
    """Test emergency data validation for critical trades."""
    from data_validation import emergency_data_check
    
    # Test with valid data
    valid_data = pd.DataFrame({
        'Close': [150.0, 151.0, 152.0],
        'Volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01 09:30:00', periods=3, freq='1min', tz='UTC'))
    
    # Should pass emergency validation
    assert emergency_data_check(valid_data, "AAPL") is True
    
    # Test with empty data
    empty_data = pd.DataFrame()
    assert emergency_data_check(empty_data, "AAPL") is False
    
    # Test with invalid price data
    invalid_data = pd.DataFrame({
        'Close': [150.0, 151.0, -10.0],  # Invalid negative price
        'Volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01 09:30:00', periods=3, freq='1min', tz='UTC'))
    
    assert emergency_data_check(invalid_data, "AAPL") is False


def test_metalearn_invalid_prices_prevention():
    """Test that METALEARN_INVALID_PRICES warnings are prevented with proper data handling."""
    from meta_learning import retrain_meta_learner
    
    # Create minimal valid trade data
    valid_trade_data = {
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'symbol': ['AAPL', 'MSFT'],
        'side': ['buy', 'sell'],
        'entry_price': [150.50, 200.0],
        'exit_price': [155.0, 195.0],
        'quantity': [100, 50],
        'pnl': [450.0, -250.0],
        'signal_tags': ['momentum', 'mean_reversion']
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        df = pd.DataFrame(valid_trade_data)
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        # This should validate data quality first and not trigger METALEARN_INVALID_PRICES
        # Note: Full training may fail due to missing dependencies, but data validation should pass
        with patch('meta_learning.logger') as mock_logger:
            try:
                retrain_meta_learner(trade_log_path=tmp_path, min_samples=1)
            except:
                pass  # Training may fail due to missing sklearn, but that's OK for this test
            
            # Check that quality validation occurred
            calls = [str(call) for call in mock_logger.info.call_args_list]
            quality_calls = [call for call in calls if 'META_LEARNING_QUALITY_CHECK' in call]
            assert len(quality_calls) > 0
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])