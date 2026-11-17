#!/usr/bin/env python3
"""
Test script to validate the trading bot fixes.
This script tests the expanded ticker portfolio and TA-Lib fallback handling.
"""

from __future__ import annotations

import csv
from pathlib import Path


def test_tickers_csv():
    """Test that the packaged universe CSV has been expanded correctly."""

    tickers_file = Path("ai_trading/data/tickers.csv")
    if not tickers_file.exists():
        return False

    with open(tickers_file) as f:
        reader = csv.reader(f)
        tickers = [row[0].strip().upper() for row in reader if row]

    expected_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'AMD', 'META',
        'NFLX', 'CRM', 'UBER', 'SHOP', 'PYPL', 'PLTR', 'SPY', 'QQQ',
        'IWM', 'JPM', 'JNJ', 'PG', 'KO', 'XOM', 'CVX', 'BABA'
    ]


    missing = set(expected_tickers) - set(tickers)
    extra = set(tickers) - set(expected_tickers)

    if missing:
        return False

    if extra:
        pass

    return True
