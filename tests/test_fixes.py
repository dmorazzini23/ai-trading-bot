#!/usr/bin/env python3
"""
Test script to validate the trading bot fixes.
This script tests the expanded ticker portfolio and TA-Lib fallback handling.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pandas_types

from ai_trading.core import bot_engine


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

        from ai_trading.strategies import imports as strategy_imports

        ta = strategy_imports.get_ta()

        # Test that ta object is always available (real or mock)
        if not hasattr(ta, "trend"):
            return False
        if not hasattr(ta, "momentum"):
            return False
        if not hasattr(ta, "volatility"):
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
        except (AttributeError, ValueError):
            pass

        return True

    except ImportError:
        return False

def _make_sample_df(
    pandas_module: Any, *, rows: int = 60, volume: int = 150_000
) -> pandas_types.DataFrame:
    base = pandas_module.Series(range(rows), dtype="float64") + 100.0
    data = {
        "open": base - 0.5,
        "high": base + 0.5,
        "low": base - 1.0,
        "close": base,
        "volume": [volume] * rows,
    }
    index = pandas_module.date_range(datetime(2024, 1, 1), periods=rows, freq="D")
    return pandas_module.DataFrame(data, index=index)


def test_screen_universe_logging(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Test that screen_universe reports accurate counters for mixed outcomes."""

    pd = pytest.importorskip("pandas")

    class DummyTA:
        _failed = False

        @staticmethod
        def atr(high, low, close, length):  # noqa: D401
            return pd.Series([1.0] * len(close))

    class DummyFetcher:
        def __init__(self, frames: dict[str, "pd.DataFrame"]):
            self.frames = frames

        def get_daily_df(self, runtime, symbol: str):  # noqa: D401, ANN001
            return self.frames.get(symbol)

    class DummyRuntime:
        def __init__(self, frames: dict[str, "pd.DataFrame"]):
            self.data_fetcher = DummyFetcher(frames)

    frames = {
        "SPY": _make_sample_df(pd),
        "VALID": _make_sample_df(pd),
        "EMPTY": _make_sample_df(pd, rows=10),
        "LOW": _make_sample_df(pd, volume=90_000),
    }

    runtime = DummyRuntime(frames)

    monkeypatch.setattr(bot_engine, "ta", DummyTA(), raising=False)
    monkeypatch.setattr(bot_engine.time, "sleep", lambda *_, **__: None)
    bot_engine._SCREEN_CACHE.clear()
    bot_engine._screening_in_progress = False

    caplog.set_level(logging.INFO)

    selected = bot_engine.screen_universe(["VALID", "EMPTY", "LOW"], runtime)

    assert selected == ["VALID"]

    summary_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("SCREEN_SUMMARY |")
    ]
    assert summary_records, "Expected SCREEN_SUMMARY log entry"
    summary = summary_records[-1]

    assert summary.tried == 3
    assert summary.valid == 1
    assert summary.empty == 1
    assert summary.failed == 1

    assert (
        summary.getMessage()
        == "SCREEN_SUMMARY | symbols=3 passed=1 failed=2"
    ), "Summary log message did not match expected format"

    assert "[SCREEN_UNIVERSE] Starting screening" in caplog.text
    assert "Selected 1 of" in caplog.text

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
