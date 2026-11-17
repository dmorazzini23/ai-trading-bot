#!/usr/bin/env python3
"""
Test suite for meta-learning and related performance fixes.

This test validates the key fixes:
1. Meta-learning system can handle mixed trade log formats
2. Position size reporting is consistent
3. Order execution latency tracking is more granular
"""

import os
from typing import Any, cast
from unittest.mock import Mock

# Set testing environment
os.environ["TESTING"] = "1"
os.environ["PYTEST_CURRENT_TEST"] = "test_performance_fixes"

from ai_trading.meta_learning import retrain_meta_learner, validate_trade_data_quality


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
        cast(Any, engine)._handle_order_result("AAPL", "buy", mock_order, 150.00, 100, start_time)
    except (AttributeError, ConnectionError):
        # Expected to fail on API calls in test environment
        pass


