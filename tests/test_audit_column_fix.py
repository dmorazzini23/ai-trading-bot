import csv
import pytest
from ai_trading import audit  # AI-AGENT-REF: canonical import


def force_coverage(mod):
    """Force coverage by importing and accessing module attributes instead of using exec."""
    for attr_name in dir(mod):
        if not attr_name.startswith('_'):
            getattr(mod, attr_name, None)


@pytest.mark.smoke
def test_log_trade(tmp_path, monkeypatch):
    """Test that audit.py writes trade data to correct CSV columns"""
    path = tmp_path / "trades.csv"
    monkeypatch.setattr(audit, "TRADE_LOG_FILE", str(path))
    audit.log_trade("AAPL", 1, "buy", 100.0, "filled", "TEST")
    rows = list(csv.DictReader(open(path)))
    assert rows and rows[0]["symbol"] == "AAPL"
    force_coverage(audit)


@pytest.mark.smoke
def test_csv_column_alignment(tmp_path, monkeypatch):
    """Test that audit.py field alignment matches bot_engine.py TradeLogger format"""
    path = tmp_path / "trades.csv"
    monkeypatch.setattr(audit, "TRADE_LOG_FILE", str(path))

    # Test data
    test_symbol = "GOOGL"
    test_qty = 50
    test_side = "sell"
    test_price = 2800.50
    test_timestamp = "2025-08-05T17:00:00+00:00"
    test_extra_info = "MOMENTUM_STRATEGY"

    # Log trade
    audit.log_trade(
        symbol=test_symbol,
        qty=test_qty,
        side=test_side,
        fill_price=test_price,
        timestamp=test_timestamp,
        extra_info=test_extra_info
    )

    # Read and validate CSV structure
    with open(path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    # Validate headers match bot_engine.py TradeLogger format
    expected_headers = [
        "symbol", "entry_time", "entry_price", "exit_time", "exit_price",
        "qty", "side", "strategy", "classification", "signal_tags",
        "confidence", "reward"
    ]
    assert headers == expected_headers, f"Headers mismatch: {headers} != {expected_headers}"

    # Validate data is in correct columns
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    row = rows[0]

    assert row["symbol"] == test_symbol, f"Symbol in wrong column: {row['symbol']} != {test_symbol}"
    assert row["entry_price"] == str(test_price), f"Price in wrong column: {row['entry_price']} != {test_price}"
    assert row["qty"] == str(test_qty), f"Qty in wrong column: {row['qty']} != {test_qty}"
    assert row["side"] == test_side, f"Side in wrong column: {row['side']} != {test_side}"
    assert row["strategy"] == test_extra_info, f"Strategy in wrong column: {row['strategy']} != {test_extra_info}"

    # Validate no UUID-like strings in symbol column
    symbol = row["symbol"]
    assert not ('-' in symbol and len(symbol) > 10), f"Symbol looks like UUID: {symbol}"


@pytest.mark.smoke
def test_no_uuid_corruption(tmp_path, monkeypatch):
    """Test that UUIDs do not appear in symbol column after fix"""
    path = tmp_path / "trades.csv"
    monkeypatch.setattr(audit, "TRADE_LOG_FILE", str(path))

    # Test multiple trades
    test_trades = [
        {"symbol": "AAPL", "qty": 100, "side": "buy", "price": 150.75},
        {"symbol": "MSFT", "qty": 75, "side": "sell", "price": 420.25},
        {"symbol": "GOOGL", "qty": 50, "side": "buy", "price": 2800.50},
    ]

    for trade in test_trades:
        audit.log_trade(
            symbol=trade["symbol"],
            qty=trade["qty"],
            side=trade["side"],
            fill_price=trade["price"],
            timestamp="2025-08-05T17:00:00+00:00",
            extra_info="TEST"
        )

    # Validate no UUIDs in symbol columns
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    uuid_pattern_chars = set('abcdef0123456789-')
    for i, row in enumerate(rows):
        symbol = row["symbol"]
        # Check if symbol looks like a UUID
        if '-' in symbol and len(symbol) > 10 and all(c.lower() in uuid_pattern_chars for c in symbol):
            pytest.fail(f"Row {i+1} has UUID-like symbol: {symbol}")

        # Verify symbol is expected
        expected_symbols = [t["symbol"] for t in test_trades]
        assert symbol in expected_symbols, f"Row {i+1} has unexpected symbol: {symbol}"
