"""Test TA-Lib enforcement and audit file creation improvements."""

import csv

from tests.mocks.app_mocks import MockConfig


def test_ta_lazy_import(monkeypatch, caplog):
    """TA library loads lazily and updates availability flag."""
    import importlib
    import ai_trading.strategies.imports as imports

    # Ensure fresh state
    importlib.reload(imports)
    assert imports.TA_AVAILABLE is False

    with caplog.at_level("INFO"):
        ta = imports.get_ta()

    assert imports.TA_AVAILABLE is True
    assert hasattr(ta, "trend")
    assert any(
        "TA library loaded successfully" in message for message in caplog.messages
    )


def test_audit_file_creation_and_permissions(tmp_path, monkeypatch):
    """Test that audit.py creates trade log file with proper permissions."""
    import sys

    # Mock config to use temporary path
    trade_log_path = tmp_path / "data" / "trades.csv"

    # Create mock config module
    # Temporarily replace config module
    original_config = sys.modules.get("config")
    sys.modules["config"] = MockConfig()

    try:
        # Import audit after mocking config
        if "ai_trading.audit" in sys.modules:
            del sys.modules["ai_trading.audit"]
        if "audit" in sys.modules:
            del sys.modules["audit"]
        # Ensure audit writes to our tmp path (data/trades.csv)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADE_LOG_FILE", "data/trades.csv")
        from ai_trading import audit  # AI-AGENT-REF: canonical import

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
            exposure=0.1,
        )

        # Verify directory was created
        assert trade_log_path.parent.exists()

        # Verify file was created
        assert trade_log_path.exists()

        # Verify file permissions (0o600)
        file_stat = trade_log_path.stat()
        file_mode = oct(file_stat.st_mode)[-3:]
        assert file_mode == "600", f"Expected file permissions 600, got {file_mode}"

        # Verify file contents
        with open(trade_log_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["symbol"] == "TEST"
        assert rows[0]["side"] == "buy"
        assert rows[0]["qty"] == "10"
        assert rows[0]["price"] == "100.0"
        assert rows[0]["exposure"] == "0.1"
        assert rows[0]["mode"] == "TEST_MODE"

        # Verify CSV header exists
        with open(trade_log_path) as f:
            first_line = f.readline().strip()
            expected_headers = "id,timestamp,symbol,side,qty,price,exposure,mode,result"
            assert first_line == expected_headers

    finally:
        # Restore original config module
        if original_config is not None:
            sys.modules["config"] = original_config
        elif "config" in sys.modules:
            del sys.modules["config"]

        # Clean up audit module
        if "ai_trading.audit" in sys.modules:
            del sys.modules["ai_trading.audit"]
        if "audit" in sys.modules:
            del sys.modules["audit"]


def test_audit_file_multiple_trades(tmp_path, monkeypatch):
    """Test that multiple trades are appended correctly without duplicate headers."""
    import sys

    trade_log_path = tmp_path / "trades.csv"

    original_config = sys.modules.get("config")
    sys.modules["config"] = MockConfig()

    try:
        if "ai_trading.audit" in sys.modules:
            del sys.modules["ai_trading.audit"]
        if "audit" in sys.modules:
            del sys.modules["audit"]
        # Ensure audit writes to our tmp path
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TRADE_LOG_FILE", "trades.csv")
        from ai_trading import audit  # AI-AGENT-REF: canonical import

        # Log first trade
        audit.log_trade("AAPL", 5, "buy", 150.0, "2024-01-01T10:00:00Z", "TEST_MODE")

        # Log second trade
        audit.log_trade("MSFT", 3, "sell", 250.0, "2024-01-01T11:00:00Z", "TEST_MODE")

        # Verify both trades are in file
        from pathlib import Path as _P
        env_path = _P(str(trade_log_path))
        try:
            env_path = _P(str(__import__('os').environ.get('TRADE_LOG_FILE', env_path)))
        except Exception:
            pass
        with open(env_path) as f:
            content = f.read()

        # Should have header + 2 data rows
        lines = content.strip().split("\n")
        assert (
            len(lines) == 3
        ), f"Expected 3 lines (header + 2 trades), got {len(lines)}"

        # Verify header appears only once
        header_count = content.count(
            "id,timestamp,symbol,side,qty,price,exposure,mode,result"
        )
        assert header_count == 1, f"Expected 1 header, found {header_count}"

        # Verify trade data
        with open(env_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["symbol"] == "AAPL"
        assert rows[1]["symbol"] == "MSFT"

    finally:
        if original_config is not None:
            sys.modules["config"] = original_config
        elif "config" in sys.modules:
            del sys.modules["config"]
        if "audit" in sys.modules:
            del sys.modules["audit"]
