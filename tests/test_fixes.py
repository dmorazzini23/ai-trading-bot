#!/usr/bin/env python3
"""
Test script to validate the trading bot fixes.
This script tests the expanded ticker portfolio and TA-Lib fallback handling.
"""

import csv
import os
import sys
from pathlib import Path


def test_tickers_csv():
    """Test that tickers.csv has been expanded correctly."""

    tickers_file = Path("tickers.csv")
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

def test_talib_imports():
    """Test TA-Lib imports and fallback handling."""

    try:
        # Set dummy environment variables to avoid config errors
        os.environ.setdefault('ALPACA_API_KEY', 'dummy')
        os.environ.setdefault('ALPACA_SECRET_KEY', 'dummy')
        os.environ.setdefault('ALPACA_BASE_URL', 'paper')
        os.environ.setdefault('WEBHOOK_SECRET', 'dummy')
        os.environ.setdefault('FLASK_PORT', '5000')

        from ai_trading.strategies.imports import TA_AVAILABLE, ta

        # Test that ta object is always available (real or mock)
        if hasattr(ta, 'trend'):
            pass
        else:
            return False

        if hasattr(ta, 'momentum'):
            pass
        else:
            return False

        if hasattr(ta, 'volatility'):
            pass
        else:
            return False

        # Test basic functionality with small dataset
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3  # 30 data points
        try:
            import pytest
            pd = pytest.importorskip("pandas")
            test_series = pd.Series(test_data)
            sma_result = ta.trend.sma_indicator(test_series, window=10)
            if sma_result is not None and len(sma_result) == len(test_data):
                pass
            else:
                pass
        except (AttributeError, ValueError):
            pass

        return True

    except ImportError:
        return False

def test_screen_universe_logging():
    """Test that screen_universe function has enhanced logging."""
    bot_engine_path = Path('ai_trading/core/bot_engine.py')
    content = bot_engine_path.read_text()

    # Check for enhanced logging statements
    if 'logger.info(f"[SCREEN_UNIVERSE] Starting screening of' in content:
        pass
    else:
        return False

    if 'filtered_out[sym] = "no_data"' in content:
        pass
    else:
        return False

    if 'f"[SCREEN_UNIVERSE] Selected {len(selected)} of {len(cand_set)} candidates' in content:
        pass
    else:
        return False

    return True

def main():
    """Run all tests."""

    tests = [
        test_tickers_csv,
        test_talib_imports,
        test_screen_universe_logging,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except (RuntimeError, OSError, ValueError):
            results.append(False)


    if all(results):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
