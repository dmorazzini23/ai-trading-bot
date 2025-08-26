from tests.optdeps import require
require("pandas")
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("alpaca_trade_api")

# AI-AGENT-REF: Replaced unsafe _raise_dynamic_exec_disabled() with direct import from shim module
from ai_trading.core.bot_engine import prepare_indicators


def test_prepare_indicators_missing_close_column():
    df = pd.DataFrame({'open': [1, 2], 'high': [1, 2], 'low': [1, 2]})
    with pytest.raises(KeyError):
        prepare_indicators(df)


def test_prepare_indicators_non_numeric_close(monkeypatch):
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
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        prepare_indicators(df)


def test_prepare_indicators_single_row():
    df = pd.DataFrame({
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [100.5],
        'volume': [1000]
    })
    result = prepare_indicators(df.copy())
    assert result.empty
