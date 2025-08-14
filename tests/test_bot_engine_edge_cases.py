import numpy as np
import pandas as pd
import pytest

# AI-AGENT-REF: Replaced unsafe _raise_dynamic_exec_disabled() with direct import from shim module
from ai_trading.core.bot_engine import prepare_indicators


def test_prepare_indicators_missing_close_column():
    print('Testing prepare_indicators with missing Close column')
    df = pd.DataFrame({'open': [1, 2], 'high': [1, 2], 'low': [1, 2]})
    with pytest.raises(KeyError):
        prepare_indicators(df)


def test_prepare_indicators_non_numeric_close(monkeypatch):
    print('Testing prepare_indicators with non-numeric Close column')
    from ai_trading.core import bot_engine

    def fake_rsi(close, length=14):
        if not pd.api.types.is_numeric_dtype(close):
            raise TypeError('close column must be numeric')
        return pd.Series(np.arange(len(close)), dtype=float)

    monkeypatch.setattr(bot_engine.ta, 'rsi', fake_rsi)
    df = pd.DataFrame({'open': [1, 2], 'high': [1, 2], 'low': [1, 2], 'close': ['a', 'b']})
    with pytest.raises(TypeError):
        prepare_indicators(df)


def test_prepare_indicators_empty_dataframe():
    print('Testing prepare_indicators with empty DataFrame')
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        prepare_indicators(df)


def test_prepare_indicators_single_row():
    print('Testing prepare_indicators with single row DataFrame')
    df = pd.DataFrame({
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [1000]
    })
    result = prepare_indicators(df.copy())
    assert result.empty
