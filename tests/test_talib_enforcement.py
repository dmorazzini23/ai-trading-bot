"""Test TA-Lib enforcement and audit file creation improvements."""

import csv
import os
import tempfile
from pathlib import Path

import pytest


def test_talib_import_enforcement():
    """Test that TA library import gracefully handles missing dependency."""
    # Read the imports file to test the TA library section
    imports_file = Path(__file__).parent.parent / "ai_trading" / "strategies" / "imports.py"
    
    with open(imports_file, 'r') as f:
        content = f.read()
    
    # Find the TA library section
    lines = content.split('\n')
    ta_start = None
    ta_end = None
    
    for i, line in enumerate(lines):
        if '# TA library for optimized technical analysis' in line:
            ta_start = i
        elif ta_start is not None and 'ta = MockTa()' in line:
            ta_end = i + 1
            break
    
    assert ta_start is not None, "Could not find TA library section"
    assert ta_end is not None, "Could not find end of TA library section"
    
    # Verify the fallback implementation exists
    assert 'TA_AVAILABLE = False' in content
    assert 'class MockTa:' in content
    assert 'TA library not available - using fallback implementation' in content
    
    # Test that the import works without raising an error
    try:
        from ai_trading.strategies.imports import ta, TA_AVAILABLE
        assert TA_AVAILABLE is True, "Expected TA_AVAILABLE to be True since ta library is installed"
        assert hasattr(ta, 'trend'), "Expected ta to have trend module"
        assert hasattr(ta, 'momentum'), "Expected ta to have momentum module"
        assert hasattr(ta, 'volatility'), "Expected ta to have volatility module"
        print("✅ TA library import successful")
    except ImportError as e:
        pytest.fail(f"TA library import should not raise ImportError with fallback: {e}")


def test_audit_file_creation_and_permissions(tmp_path, monkeypatch):
    """Test that audit.py creates trade log file with proper permissions."""
    import sys
    
    # Mock config to use temporary path
    trade_log_path = tmp_path / "data" / "trades.csv"
    
    # Create mock config module
    class MockConfig:
        TRADE_LOG_FILE = str(trade_log_path)
        TRADE_AUDIT_DIR = str(tmp_path / "audit")
    
    # Temporarily replace config module
    original_config = sys.modules.get('config')
    sys.modules['config'] = MockConfig()
    
    try:
        # Import audit after mocking config
        if 'audit' in sys.modules:
            del sys.modules['audit']
        import audit
        
        # Ensure the file doesn't exist initially
        assert not trade_log_path.exists()
        assert not trade_log_path.parent.exists()
        
        # Call log_trade which should create the directory and file
        audit.log_trade(
            symbol="TEST",
            qty=10,
            side="buy", 
            fill_price=100.0,
            timestamp="2024-01-01T10:00:00Z",
            extra_info="TEST_MODE",
            exposure=0.1
        )
        
        # Verify directory was created
        assert trade_log_path.parent.exists()
        
        # Verify file was created
        assert trade_log_path.exists()
        
        # Verify file permissions (0o664)
        file_stat = trade_log_path.stat()
        file_mode = oct(file_stat.st_mode)[-3:]
        assert file_mode == "664", f"Expected file permissions 664, got {file_mode}"
        
        # Verify file contents
        with open(trade_log_path, 'r') as f:
            rows = list(csv.DictReader(f))
            
        assert len(rows) == 1
        assert rows[0]['symbol'] == 'TEST'
        assert rows[0]['side'] == 'buy'
        assert rows[0]['qty'] == '10'
        assert rows[0]['price'] == '100.0'
        assert rows[0]['exposure'] == '0.1'
        assert rows[0]['mode'] == 'TEST_MODE'
        
        # Verify CSV header exists
        with open(trade_log_path, 'r') as f:
            first_line = f.readline().strip()
            expected_headers = "id,timestamp,symbol,side,qty,price,exposure,mode,result"
            assert first_line == expected_headers
            
    finally:
        # Restore original config module
        if original_config is not None:
            sys.modules['config'] = original_config
        elif 'config' in sys.modules:
            del sys.modules['config']
        
        # Clean up audit module
        if 'audit' in sys.modules:
            del sys.modules['audit']


def test_audit_file_multiple_trades(tmp_path, monkeypatch):
    """Test that multiple trades are appended correctly without duplicate headers."""
    import sys
    
    trade_log_path = tmp_path / "trades.csv"
    
    class MockConfig:
        TRADE_LOG_FILE = str(trade_log_path)
        TRADE_AUDIT_DIR = str(tmp_path)
    
    original_config = sys.modules.get('config')
    sys.modules['config'] = MockConfig()
    
    try:
        if 'audit' in sys.modules:
            del sys.modules['audit']
        import audit
        
        # Log first trade
        audit.log_trade("AAPL", 5, "buy", 150.0, "2024-01-01T10:00:00Z")
        
        # Log second trade  
        audit.log_trade("MSFT", 3, "sell", 250.0, "2024-01-01T11:00:00Z")
        
        # Verify both trades are in file
        with open(trade_log_path, 'r') as f:
            content = f.read()
            
        # Should have header + 2 data rows
        lines = content.strip().split('\n')
        assert len(lines) == 3, f"Expected 3 lines (header + 2 trades), got {len(lines)}"
        
        # Verify header appears only once
        header_count = content.count("id,timestamp,symbol,side,qty,price,exposure,mode,result")
        assert header_count == 1, f"Expected 1 header, found {header_count}"
        
        # Verify trade data
        with open(trade_log_path, 'r') as f:
            rows = list(csv.DictReader(f))
            
        assert len(rows) == 2
        assert rows[0]['symbol'] == 'AAPL'
        assert rows[1]['symbol'] == 'MSFT'
        
    finally:
        if original_config is not None:
            sys.modules['config'] = original_config
        elif 'config' in sys.modules:
            del sys.modules['config']
        if 'audit' in sys.modules:
            del sys.modules['audit']