import sys
import tempfile
import os
from pathlib import Path
import pytest

# Mock the config module to avoid environment variable requirements
class MockConfig:
    TRADE_LOG_FILE = 'logs/trades.csv'

sys.modules['config'] = MockConfig()

import meta_learning

def test_trigger_meta_learning_conversion_pure_meta_format():
    """Test trigger function with pure meta-learning format - should return True immediately."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write meta-learning format data
        f.write("symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n")
        f.write("TEST,2025-08-05T23:17:35Z,100.0,2025-08-05T23:18:35Z,105.0,10,buy,test_strategy,test,signal1+signal2,0.8,5.0\n")
        f.write("AAPL,2025-08-05T23:19:35Z,150.0,2025-08-05T23:20:35Z,155.0,5,buy,test_strategy,test,signal3,0.7,25.0\n")
        f.write("MSFT,2025-08-05T23:21:35Z,300.0,2025-08-05T23:22:35Z,295.0,2,sell,test_strategy,test,signal4,0.6,-10.0\n")
        test_file = f.name
    
    try:
        # Set the trade log file path
        MockConfig.TRADE_LOG_FILE = test_file
        
        # Test trade data
        test_trade = {
            'symbol': 'TEST', 
            'qty': 10, 
            'side': 'buy', 
            'price': 100.0, 
            'timestamp': '2025-08-05T23:17:35Z', 
            'order_id': 'test-001', 
            'status': 'filled'
        }
        
        # Verify quality report shows pure meta format
        quality_report = meta_learning.validate_trade_data_quality(test_file)
        assert quality_report['mixed_format_detected'] is False
        assert quality_report['audit_format_rows'] == 0
        assert quality_report['meta_format_rows'] > 0
        
        # Test the trigger function - should return True immediately (no conversion needed)
        result = meta_learning.trigger_meta_learning_conversion(test_trade)
        assert result is True
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_trigger_meta_learning_conversion_pure_audit_format():
    """Test trigger function with pure audit format - should attempt conversion."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write audit format data
        f.write("order_id,timestamp,symbol,side,qty,price,mode,status\n")
        f.write("123e4567-e89b-12d3-a456-426614174000,2025-08-05T23:17:35Z,TEST,buy,10,100.0,live,filled\n")
        f.write("234e5678-e89b-12d3-a456-426614174001,2025-08-05T23:18:35Z,TEST,sell,10,105.0,live,filled\n")
        f.write("345e6789-e89b-12d3-a456-426614174002,2025-08-05T23:19:35Z,AAPL,buy,5,150.0,live,filled\n")
        test_file = f.name
    
    try:
        # Set the trade log file path
        MockConfig.TRADE_LOG_FILE = test_file
        
        # Test trade data
        test_trade = {
            'symbol': 'TEST', 
            'qty': 10, 
            'side': 'buy', 
            'price': 100.0, 
            'timestamp': '2025-08-05T23:17:35Z', 
            'order_id': 'test-001', 
            'status': 'filled'
        }
        
        # Verify quality report shows pure audit format
        quality_report = meta_learning.validate_trade_data_quality(test_file)
        assert quality_report['mixed_format_detected'] is False
        assert quality_report['audit_format_rows'] > 0
        assert quality_report['meta_format_rows'] == 0
        
        # Test the trigger function - should attempt conversion and return True if successful
        result = meta_learning.trigger_meta_learning_conversion(test_trade)
        assert result is True  # Should succeed in conversion
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_trigger_meta_learning_conversion_mixed_format():
    """Test trigger function with mixed format - should attempt conversion."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write mixed format data (meta headers with audit data)
        f.write("symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n")
        f.write("123e4567-e89b-12d3-a456-426614174000,2025-08-05T23:17:35Z,TEST,buy,10,100.0,live,filled\n")
        f.write("234e5678-e89b-12d3-a456-426614174001,2025-08-05T23:18:35Z,TEST,sell,10,105.0,live,filled\n")
        test_file = f.name
    
    try:
        # Set the trade log file path
        MockConfig.TRADE_LOG_FILE = test_file
        
        # Test trade data
        test_trade = {
            'symbol': 'TEST', 
            'qty': 10, 
            'side': 'buy', 
            'price': 100.0, 
            'timestamp': '2025-08-05T23:17:35Z', 
            'order_id': 'test-001', 
            'status': 'filled'
        }
        
        # Verify quality report shows mixed format
        quality_report = meta_learning.validate_trade_data_quality(test_file)
        assert quality_report['mixed_format_detected'] is True
        
        # Test the trigger function - should attempt conversion and return True if successful
        result = meta_learning.trigger_meta_learning_conversion(test_trade)
        assert result is True  # Should succeed in conversion
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_trigger_meta_learning_conversion_missing_file():
    """Test trigger function with missing file - should return False."""
    # Set a non-existent file path
    MockConfig.TRADE_LOG_FILE = '/tmp/non_existent_file.csv'
    
    test_trade = {
        'symbol': 'TEST', 
        'qty': 10, 
        'side': 'buy', 
        'price': 100.0, 
        'timestamp': '2025-08-05T23:17:35Z', 
        'order_id': 'test-001', 
        'status': 'filled'
    }
    
    # Test the trigger function - should return False for missing file
    result = meta_learning.trigger_meta_learning_conversion(test_trade)
    assert result is False


def test_trigger_meta_learning_conversion_problem_statement_exact():
    """Test the exact scenario from the problem statement."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create exactly the scenario: mixed_format_detected=False, audit_format_rows=0, meta_format_rows=4
        f.write("symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n")
        f.write("TEST,2025-08-05T23:17:35Z,100.0,2025-08-05T23:18:35Z,105.0,10,buy,test_strategy,test,signal1+signal2,0.8,5.0\n")
        f.write("AAPL,2025-08-05T23:19:35Z,150.0,2025-08-05T23:20:35Z,155.0,5,buy,test_strategy,test,signal3,0.7,25.0\n")
        f.write("MSFT,2025-08-05T23:21:35Z,300.0,2025-08-05T23:22:35Z,295.0,2,sell,test_strategy,test,signal4,0.6,-10.0\n")
        f.write("GOOGL,2025-08-05T23:23:35Z,2500.0,2025-08-05T23:24:35Z,2505.0,1,buy,test_strategy,test,signal5,0.9,5.0\n")
        test_file = f.name
    
    try:
        MockConfig.TRADE_LOG_FILE = test_file
        
        test_trade = {
            'symbol': 'TEST', 
            'qty': 10, 
            'side': 'buy', 
            'price': 100.0, 
            'timestamp': '2025-08-05T23:17:35Z', 
            'order_id': 'test-001', 
            'status': 'filled'
        }
        
        # Verify we have the exact scenario from problem statement
        quality_report = meta_learning.validate_trade_data_quality(test_file)
        assert quality_report['mixed_format_detected'] is False
        assert quality_report['audit_format_rows'] == 0
        assert quality_report['meta_format_rows'] > 0  # Should be 5 (4 data + 1 header)
        
        # This should return True immediately (no conversion needed)
        result = meta_learning.trigger_meta_learning_conversion(test_trade)
        assert result is True, "Should return True for properly formatted meta-learning files"
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)