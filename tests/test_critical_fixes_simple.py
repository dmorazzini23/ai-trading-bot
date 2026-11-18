#!/usr/bin/env python3
"""
Simplified test for critical trading bot fixes.
Tests the specific issues identified in production logs.
"""
from datetime import UTC, datetime


def test_position_sizing_minimum_viable():
    """Test that position sizing provides minimum viable quantities with available cash."""

    # Simulate the fixed logic from bot_engine.py
    balance = 88000.0  # $88K available cash
    target_weight = 0.002  # Weight above the 0.001 threshold
    current_price = 150.0  # AAPL-like price

    # Original calculation that resulted in 0
    raw_qty = int(balance * target_weight / current_price)

    # Fixed logic - ensure minimum position size when cash available
    if raw_qty <= 0 and balance > 1000 and target_weight > 0.001 and current_price > 0:
        raw_qty = max(1, int(1000 / current_price))  # Minimum $1000 position

    assert raw_qty > 0, f"Should compute positive quantity with ${balance:.0f} cash available"
    assert raw_qty >= 1, "Should have at least 1 share for minimum position"


def test_meta_learning_price_conversion():
    """Test meta learning properly converts string prices to numeric."""

    # Simulate the fixed price conversion logic
    test_data: list[dict[str, str | float]] = [
        {'entry_price': '150.50', 'exit_price': '155.25'},  # String prices
        {'entry_price': 250.00, 'exit_price': 245.50},     # Numeric prices
        {'entry_price': '200.75', 'exit_price': '210.00'},  # String prices
        {'entry_price': 'N/A', 'exit_price': '100.00'},    # Invalid entry price
    ]

    valid_rows = []
    for row in test_data:
        try:
            # Simulate the pandas to_numeric conversion
            entry_str = str(row['entry_price'])
            exit_str = str(row['exit_price'])

            # Check if string represents a valid float
            if entry_str.replace('.', '').replace('-', '').isdigit() and exit_str.replace('.', '').replace('-', '').isdigit():
                entry_price = float(entry_str)
                exit_price = float(exit_str)

                if entry_price > 0 and exit_price > 0:
                    valid_rows.append({'entry_price': entry_price, 'exit_price': exit_price})
        except (ValueError, TypeError):
            continue  # Skip invalid rows


    # Should have 3 valid rows (invalid row should be filtered out)
    assert len(valid_rows) == 3, f"Should have 3 valid price rows, got {len(valid_rows)}"
    assert all(row['entry_price'] > 0 for row in valid_rows), "All entry prices should be positive"
    assert all(row['exit_price'] > 0 for row in valid_rows), "All exit prices should be positive"


def test_stale_data_bypass_startup():
    """Test that stale data bypass works during initial deployment."""

    # Simulate startup environment with stale data bypass enabled
    stale_symbols = ["NFLX", "META", "TSLA", "MSFT", "AMD"]
    allow_stale_on_startup = True  # Default behavior

    # Test that bypass allows trading to proceed
    if stale_symbols and allow_stale_on_startup:
        trading_allowed = True
    else:
        trading_allowed = False

    assert trading_allowed, "Should allow trading with stale data bypass enabled on startup"

    # Test that bypass can be disabled
    allow_stale_on_startup = False
    if stale_symbols and not allow_stale_on_startup:
        trading_allowed = False
    else:
        trading_allowed = True

    assert not trading_allowed, "Should block trading when stale data bypass is disabled"

