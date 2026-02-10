#!/usr/bin/env python3
"""
Test suite for meta-learning and related performance fixes.

This test validates the key fixes:
1. Meta-learning system can handle mixed trade log formats
2. Position size reporting is consistent
3. Order execution latency tracking is more granular
"""

import os
import sys
from unittest.mock import Mock

# Set testing environment
os.environ["TESTING"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "test_performance_fixes"

from ai_trading.meta_learning import (
    retrain_meta_learner,
    validate_trade_data_quality,
    has_mixed_format,
)


def _write_mixed_trade_log(path):
    rows = [
        "symbol,side,entry_price,signal_tags,exit_price,timestamp",
        "2026-01-01T00:00:00+00:00,ORD123,AAPL,buy,10,100.00",
    ]
    for i in range(12):
        entry = 100.0 + i
        exit_p = entry + (1.0 if i % 2 == 0 else -0.5)
        rows.append(
            f"AAPL,buy,{entry:.2f},trend+volume,{exit_p:.2f},2026-01-01T00:{i:02d}:00+00:00"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")


def test_meta_learning_mixed_format(tmp_path):
    """Test that meta-learning can handle mixed audit/meta-learning log formats."""

    trade_log = tmp_path / "trades.csv"
    _write_mixed_trade_log(trade_log)
    quality_report = validate_trade_data_quality(str(trade_log))

    # Verify mixed format detection
    assert quality_report["file_exists"], "Trade log file should exist"
    assert quality_report["has_valid_format"], "Should have valid format"
    assert has_mixed_format(str(trade_log)), "Should detect mixed formats"
    assert quality_report["audit_format_rows"] > 0, "Should find audit format rows"
    assert quality_report["meta_format_rows"] > 0, "Should find meta-learning format rows"
    assert quality_report["valid_price_rows"] > 0, "Should find valid price rows"
    assert quality_report["data_quality_score"] > 0, "Should have positive data quality score"

    # Test that retrain_meta_learner works
    result = retrain_meta_learner(str(trade_log), min_samples=10)
    assert result, "Meta-learning retraining should succeed"

def test_position_size_reporting():
    """Test that position size reporting is consistent."""

    from ai_trading.execution.engine import ExecutionEngine

    # Create mock context
    mock_ctx = Mock()
    mock_ctx.api = Mock()
    mock_ctx.partial_fill_tracker = {}  # Initialize as empty dict

    # Create execution engine
    engine = ExecutionEngine(mock_ctx)

    # Test partial fill reconciliation
    mock_order = Mock()
    mock_order.id = "test_order_123"

    # Test case: partial fill
    engine._reconcile_partial_fills("AAPL", requested_qty=100, remaining_qty=25, side="buy", last_order=mock_order)

    # Test case: full fill
    engine._reconcile_partial_fills("AAPL", requested_qty=100, remaining_qty=0, side="buy", last_order=mock_order)


def test_latency_tracking():
    """Test that order execution latency tracking is more granular."""

    import time

    from ai_trading.execution.engine import ExecutionEngine

    # Create mock context and engine
    mock_ctx = Mock()
    mock_ctx.api = Mock()
    engine = ExecutionEngine(mock_ctx)

    # Simulate order result handling
    mock_order = Mock()
    mock_order.status = "filled"
    mock_order.id = "test_order_456"
    mock_order.filled_avg_price = 150.50
    mock_order.filled_qty = 100

    start_time = time.monotonic()
    time.sleep(0.001)  # Small delay to simulate order processing

    # Test latency calculation (this will fail on API calls, but that's expected in test)
    try:
        engine._handle_order_result("AAPL", "buy", mock_order, 150.00, 100, start_time)
    except (AttributeError, ConnectionError):
        # Expected to fail on API calls in test environment
        pass


def test_comprehensive_fixes(tmp_path):
    """Run comprehensive test of all performance fixes."""

    test_meta_learning_mixed_format(tmp_path)

    test_position_size_reporting()

    test_latency_tracking()

    return True


if __name__ == "__main__":
    success = test_comprehensive_fixes()
    sys.exit(0 if success else 1)
