import ast
import types
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Build a lightweight bot_engine module exposing only prepare_indicators
if 'bot_engine' not in sys.modules:
    src_path = Path(__file__).resolve().parents[1] / 'bot_engine.py'
    source = src_path.read_text()
    tree = ast.parse(source)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_indicators')
    mod = types.ModuleType('bot_engine')
    mod.pd = pd
    mod.np = np
    mod.ta = types.SimpleNamespace(
        rsi=lambda close, length=14: pd.Series(np.arange(len(close)), dtype=float)
    )
    exec(compile(ast.Module([func], []), filename=str(src_path), mode='exec'), mod.__dict__)
    sys.modules['bot_engine'] = mod

from bot_engine import prepare_indicators


def test_prepare_indicators_missing_close_column():
    print('Testing prepare_indicators with missing Close column')
    df = pd.DataFrame({'open': [1, 2], 'high': [1, 2], 'low': [1, 2]})
    with pytest.raises(KeyError):
        prepare_indicators(df)


def test_prepare_indicators_non_numeric_close(monkeypatch):
    print('Testing prepare_indicators with non-numeric Close column')
    import bot_engine

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
