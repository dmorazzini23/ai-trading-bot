from tests.optdeps import require
require("pandas")
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alpaca_trade_api")

# AI-AGENT-REF: Replaced unsafe _raise_dynamic_exec_disabled() with direct import from shim module
from ai_trading.core.bot_engine import prepare_indicators

np.random.seed(0)


def test_prepare_indicators_creates_required_columns():
    df = pd.DataFrame({
        'open': np.random.uniform(100, 200, 30),
        'high': np.random.uniform(100, 200, 30),
        'low': np.random.uniform(100, 200, 30),
        'close': np.random.uniform(100, 200, 30),
        'volume': np.random.randint(1_000_000, 5_000_000, 30)
    })

    result = prepare_indicators(df.copy())

    required = ['ichimoku_conv', 'ichimoku_base', 'stochrsi']
    for col in required:
        assert col in result.columns, f"Missing expected column: {col}"

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_prepare_indicators_insufficient_data():
    """prepare_indicators should return an empty DataFrame when there is
    insufficient historical data for rolling calculations."""

    df = pd.DataFrame({
        'open': np.random.uniform(100, 200, 5),
        'high': np.random.uniform(100, 200, 5),
        'low': np.random.uniform(100, 200, 5),
        'close': np.random.uniform(100, 200, 5),
        'volume': np.random.randint(1_000_000, 5_000_000, 5),
    })

    result = prepare_indicators(df.copy())

    assert result.empty or result.shape[0] == 0


def test_prepare_indicators_all_nan_columns():
    """prepare_indicators should drop all rows when input columns are entirely NaN."""

    df = pd.DataFrame({
        'open': [np.nan] * 30,
        'high': [np.nan] * 30,
        'low': [np.nan] * 30,
        'close': [np.nan] * 30,
        'volume': [np.nan] * 30,
    })

    from ai_trading.core import bot_engine

    original_rsi = bot_engine.ta.rsi
    bot_engine.ta.rsi = lambda close, length=14: pd.Series([np.nan] * len(close))
    try:
        result = prepare_indicators(df.copy())
    finally:
        bot_engine.ta.rsi = original_rsi

    assert result.empty or result.shape[0] == 0
