"""Test TA-Lib enforcement and audit file creation improvements."""

import csv
import os
import tempfile
from pathlib import Path

import pytest


def test_talib_import_enforcement():
    """Test that TA-Lib import raises ImportError when not available."""
    # Read the imports file to test the TA-Lib section
    imports_file = Path(__file__).parent.parent / "ai_trading" / "strategies" / "imports.py"
    
    with open(imports_file, 'r') as f:
        content = f.read()
    
    # Find the TA-Lib section
    lines = content.split('\n')
    talib_start = None
    talib_end = None
    
    for i, line in enumerate(lines):
        if '# TA-Lib required dependency' in line:
            talib_start = i
        elif talib_start is not None and line.startswith('except ImportError as e:'):
            # Find the end of this except block
            for j in range(i, len(lines)):
                if j + 1 < len(lines) and lines[j].strip() == '' and not lines[j+1].startswith(' ') and not lines[j+1].startswith('\t'):
                    talib_end = j
                    break
            break
    
    assert talib_start is not None, "Could not find TA-Lib import section"
    
    # Extract just the TA-Lib import code
    talib_code = '\n'.join(lines[talib_start:talib_end])
    
    # Test that it raises the expected ImportError
    with pytest.raises(ImportError) as exc_info:
        exec(talib_code)
    
    error_msg = str(exc_info.value)
    assert "TA-Lib C library not found" in error_msg
    assert "pip install TA-Lib" in error_msg
    assert "libta-lib0-dev" in error_msg


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