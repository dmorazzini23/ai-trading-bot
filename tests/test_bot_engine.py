import ast
import types
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Build a lightweight bot_engine module exposing only prepare_indicators
if 'bot_engine' not in sys.modules:
    src_path = Path(__file__).resolve().parents[1] / 'bot_engine.py'
    source = src_path.read_text()
    tree = ast.parse(source)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_indicators')
    mod = types.ModuleType('bot_engine')
    mod.pd = pd
    mod.np = np
    mod.ta = types.SimpleNamespace(rsi=lambda close, length=14: pd.Series(np.arange(len(close))))
    exec(compile(ast.Module([func], []), filename='bot_engine_stub', mode='exec'), mod.__dict__)
    sys.modules['bot_engine'] = mod

from bot_engine import prepare_indicators


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
