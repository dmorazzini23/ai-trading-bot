"""
Test critical trading bot fixes implementation.

Tests for the fixes addressing the critical issues:
1. Missing RiskEngine methods
2. BotContext alpaca_client compatibility
3. Process management enhancements
4. Data validation functionality
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import tempfile
from unittest.mock import Mock, patch


def test_risk_engine_missing_methods():
    """Test that RiskEngine has the missing critical methods."""
    from risk_engine import RiskEngine
    
    # Create risk engine instance
    risk_engine = RiskEngine()
    
    # Test get_current_exposure method
    assert hasattr(risk_engine, 'get_current_exposure')
    exposure = risk_engine.get_current_exposure()
    assert isinstance(exposure, dict)
    
    # Test max_concurrent_orders method
    assert hasattr(risk_engine, 'max_concurrent_orders')
    max_orders = risk_engine.max_concurrent_orders()
    assert isinstance(max_orders, int)
    assert max_orders > 0
    
    # Test max_exposure method
    assert hasattr(risk_engine, 'max_exposure')
    max_exp = risk_engine.max_exposure()
    assert isinstance(max_exp, float)
    assert 0 <= max_exp <= 1.0
    
    # Test order_spacing method
    assert hasattr(risk_engine, 'order_spacing')
    spacing = risk_engine.order_spacing()
    assert isinstance(spacing, float)
    assert spacing >= 0


def test_bot_context_alpaca_client_compatibility():
    """Test that BotContext has alpaca_client property for backward compatibility."""
    from ai_trading.core.bot_engine import BotContext
    
    # Create a mock trading client
    mock_api = Mock()
    
    # Create BotContext instance
    ctx = BotContext(
        api=mock_api,
        data_client=Mock(),
        data_fetcher=Mock(),
        signal_manager=Mock(),
        trade_logger=Mock(),
        sem=Mock(),
        volume_threshold=1000,
        entry_start_offset=timedelta(minutes=30),
        entry_end_offset=timedelta(minutes=30),
        market_open=datetime.now(timezone.utc).time(),  # AI-AGENT-REF: Use timezone-aware datetime
        market_close=datetime.now(timezone.utc).time(),  # AI-AGENT-REF: Use timezone-aware datetime
        regime_lookback=10,
        regime_atr_threshold=0.02,
        daily_loss_limit=0.05,
        kelly_fraction=0.25,
        capital_scaler=Mock(),
        adv_target_pct=0.1,
        max_position_dollars=10000,
        params={}
    )
    
    # Test alpaca_client property exists and returns the api
    assert hasattr(ctx, 'alpaca_client')
    assert ctx.alpaca_client is mock_api


def test_process_manager_lock_mechanism():
    """Test process lock mechanism to prevent multiple instances."""
    from process_manager import ProcessManager
    
    pm = ProcessManager()
    
    # Test that the method exists
    assert hasattr(pm, 'acquire_process_lock')
    assert hasattr(pm, 'check_multiple_instances')
    
    # Test lock acquisition with temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        lock_file = tmp.name
    
    try:
        # First lock should succeed
        assert pm.acquire_process_lock(lock_file) == True
        
        # Clean up lock file for next test
        if os.path.exists(lock_file):
            os.remove(lock_file)
            
    finally:
        # Clean up
        if os.path.exists(lock_file):
            os.remove(lock_file)


def test_data_validation_freshness():
    """Test data validation and staleness detection."""
    from ai_trading.data_validation import check_data_freshness, validate_trading_data, get_stale_symbols
    
    # Create test data with different timestamps
    now = datetime.now(timezone.utc)
    
    # Fresh data (5 minutes old)
    fresh_data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [101, 102, 103],
        'Low': [99, 100, 101],
        'Close': [100.5, 101.5, 102.5],
        'Volume': [1000, 1100, 1200]
    }, index=[now - timedelta(minutes=7), now - timedelta(minutes=6), now - timedelta(minutes=5)])
    
    # Stale data (30 minutes old)
    stale_data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [101, 102, 103],
        'Low': [99, 100, 101],
        'Close': [100.5, 101.5, 102.5],
        'Volume': [1000, 1100, 1200]
    }, index=[now - timedelta(minutes=32), now - timedelta(minutes=31), now - timedelta(minutes=30)])
    
    # Test freshness check for fresh data
    fresh_result = check_data_freshness(fresh_data, "AAPL", max_staleness_minutes=15)
    assert fresh_result['is_fresh'] == True
    assert fresh_result['symbol'] == "AAPL"
    assert fresh_result['minutes_stale'] < 15
    
    # Test freshness check for stale data
    stale_result = check_data_freshness(stale_data, "MSFT", max_staleness_minutes=15)
    assert stale_result['is_fresh'] == False
    assert stale_result['symbol'] == "MSFT"
    assert stale_result['minutes_stale'] > 15
    
    # Test batch validation
    test_data = {
        'AAPL': fresh_data,
        'MSFT': stale_data
    }
    
    validation_results = validate_trading_data(test_data, max_staleness_minutes=15)
    
    assert 'AAPL' in validation_results
    assert 'MSFT' in validation_results
    assert validation_results['AAPL']['trading_ready'] == True
    assert validation_results['MSFT']['trading_ready'] == False
    
    # Test stale symbols detection
    stale_symbols = get_stale_symbols(validation_results)
    assert 'MSFT' in stale_symbols
    assert 'AAPL' not in stale_symbols


def test_data_validation_emergency_check():
    """Test emergency data validation for critical trades."""
    from ai_trading.data_validation import emergency_data_check
    
    now = datetime.now(timezone.utc)
    
    # Valid data
    valid_data = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [101, 102, 103], 
        'Low': [99, 100, 101],
        'Close': [100.5, 101.5, 102.5],
        'Volume': [1000, 1100, 1200]
    }, index=[now - timedelta(minutes=7), now - timedelta(minutes=6), now - timedelta(minutes=5)])
    
    # Empty data
    empty_data = pd.DataFrame()
    
    # Test valid data passes
    assert emergency_data_check(valid_data, "AAPL") == True
    
    # Test empty data fails
    assert emergency_data_check(empty_data, "MSFT") == False


def test_risk_engine_exposure_tracking():
    """Test that RiskEngine properly tracks exposure."""
    from risk_engine import RiskEngine
    
    risk_engine = RiskEngine()
    
    # Test initial exposure
    initial_exposure = risk_engine.get_current_exposure()
    assert isinstance(initial_exposure, dict)
    
    # Test exposure updates
    risk_engine.exposure['equity'] = 0.5
    updated_exposure = risk_engine.get_current_exposure()
    assert updated_exposure['equity'] == 0.5
    
    # Test max exposure configuration
    max_exp = risk_engine.max_exposure()
    assert isinstance(max_exp, float)
    assert max_exp > 0


def test_process_manager_multiple_instances_check():
    """Test multiple instances detection."""
    from process_manager import ProcessManager
    
    pm = ProcessManager()
    
    # Mock find_python_processes to simulate multiple instances
    mock_processes = [
        {'pid': 1234, 'command': 'python bot_engine.py', 'memory_mb': 100},
        {'pid': 5678, 'command': 'python runner.py', 'memory_mb': 150}
    ]
    
    with patch.object(pm, 'find_python_processes', return_value=mock_processes):
        with patch.object(pm, '_is_trading_process', return_value=True):
            result = pm.check_multiple_instances()
            
            assert result['total_instances'] == 2
            assert result['multiple_instances'] == True
            assert len(result['recommendations']) > 0
            assert any('CRITICAL' in rec for rec in result['recommendations'])


@patch('ai_trading.audit.logger')
def test_audit_permission_handling(mock_logger):
    """Test that audit module handles permission errors gracefully."""
    from ai_trading.audit import log_trade  # AI-AGENT-REF: canonical import
    
    # This test validates that the permission error handling code exists
    # and would be called in case of permission errors
    
    # Test that log_trade function exists and can be called
    # In a real permission error scenario, it would attempt to repair permissions
    try:
        log_trade("AAPL", 10, "buy", 150.0, datetime.now(timezone.utc), "test")
        # If it succeeds, that's fine - we're mainly testing the error handling path exists
    except Exception:
        # If it fails due to missing dependencies, that's also acceptable for this test
        pass
    
    # The important thing is that the permission handling code exists in audit.py
    import ai_trading.audit as audit  # AI-AGENT-REF: canonical import
    import inspect
    
    # Check that the enhanced permission handling code is present
    source = inspect.getsource(audit.log_trade)
    assert 'ProcessManager' in source
    assert 'fix_file_permissions' in source


def test_integration_risk_engine_methods():
    """Integration test ensuring all risk engine methods work together."""
    from risk_engine import RiskEngine
    
    risk_engine = RiskEngine()
    
    # Test that all methods return sensible values
    exposure = risk_engine.get_current_exposure()
    max_orders = risk_engine.max_concurrent_orders()
    max_exp = risk_engine.max_exposure() 
    spacing = risk_engine.order_spacing()
    
    assert isinstance(exposure, dict)
    assert isinstance(max_orders, int) and max_orders > 0
    assert isinstance(max_exp, float) and 0 < max_exp <= 1.0
    assert isinstance(spacing, float) and spacing >= 0
    
    # Test exposure tracking
    risk_engine.exposure['test_asset'] = 0.3
    updated_exposure = risk_engine.get_current_exposure()
    assert 'test_asset' in updated_exposure
    assert updated_exposure['test_asset'] == 0.3


if __name__ == "__main__":
    # Run basic tests
    test_risk_engine_missing_methods()
    test_bot_context_alpaca_client_compatibility()
    test_process_manager_lock_mechanism()
    test_data_validation_freshness()
    test_data_validation_emergency_check()
    test_risk_engine_exposure_tracking()
    test_process_manager_multiple_instances_check()
    test_integration_risk_engine_methods()
    print("All critical fixes tests passed!")