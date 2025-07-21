import ast
import sys
import types
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

# Build lightweight module exposing unit-testable functions from bot_engine
SRC_PATH = Path(__file__).resolve().parents[1] / "bot_engine.py"
SOURCE = SRC_PATH.read_text()
TREE = ast.parse(SOURCE)
FUNC_NAMES = {
    "initialize_bot",
    "run_trading_cycle",
    "generate_signals",
    "execute_trades",
    "load_model",
    "health_check",
    "EnsembleModel",
}
FUNCS = [
    n
    for n in TREE.body
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in FUNC_NAMES
]
MOD = types.ModuleType("bot_engine_unit")
MOD.np = np
MOD.pd = pd
MOD.os = __import__("os")
MOD.utils = __import__("utils")
MOD.types = types
import joblib

MOD.joblib = joblib
import logging

MOD.logger = logging.getLogger("test")
MOD.MODEL_PATH = "model.pkl"
MOD.MODEL_RF_PATH = "model_rf.pkl"
MOD.MODEL_XGB_PATH = "model_xgb.pkl"
MOD.MODEL_LGB_PATH = "model_lgb.pkl"
exec(compile(ast.Module(FUNCS, []), filename=str(SRC_PATH), mode="exec"), MOD.__dict__)
MOD.MODEL_PATH = "model.pkl"
MOD.MODEL_RF_PATH = "model_rf.pkl"
MOD.MODEL_XGB_PATH = "model_xgb.pkl"
MOD.MODEL_LGB_PATH = "model_lgb.pkl"


# Helper dummy classes
class DummyAPI:
    def __init__(self):
        self.calls = []

    def submit_order(self, symbol, qty, side):
        self.calls.append((symbol, qty, side))


def dummy_loader():
    return pd.DataFrame({"price": [1, 2, 3]})


# --- initialize_bot ---------------------------------------------------------

def test_initialize_bot_returns_ctx_and_state():
    api = DummyAPI()
    ctx, state = MOD.initialize_bot(api, dummy_loader)
    assert hasattr(ctx, "api") and ctx.api is api
    assert hasattr(ctx, "data_loader")
    assert state is not None


# --- generate_signals -------------------------------------------------------
@pytest.mark.parametrize(
    "prices,expected",
    [
        ([1, 2, 3], [0, 1, 1]),
        ([3, 2, 1], [0, -1, -1]),
        ([1, 1, 1], [0, 0, 0]),
    ],
)
def test_generate_signals_basic(prices, expected):
    df = pd.DataFrame({"price": prices})
    result = MOD.generate_signals(df)
    assert result.tolist() == expected


def test_generate_signals_missing_column_raises():
    with pytest.raises(KeyError):
        MOD.generate_signals(pd.DataFrame({"close": [1, 2]}))


# --- execute_trades ---------------------------------------------------------

def test_execute_trades_sends_orders(monkeypatch):
    api = DummyAPI()
    ctx = types.SimpleNamespace(api=api)
    sig = pd.Series([1, 0, -1], index=["A", "B", "C"])
    orders = MOD.execute_trades(ctx, sig)
    assert orders == [("A", "buy"), ("C", "sell")]
    assert api.calls == [("A", 1, "buy"), ("C", 1, "sell")]


# --- run_trading_cycle ------------------------------------------------------

def test_run_trading_cycle_integration(monkeypatch):
    api = DummyAPI()
    ctx = types.SimpleNamespace(api=api)
    df = pd.DataFrame({"price": [1, 3, 2]}, index=["A", "B", "C"])
    orders = MOD.run_trading_cycle(ctx, df)
    assert orders == [("B", "buy"), ("C", "sell")]


# --- load_model -------------------------------------------------------------

def test_load_model_single(tmp_path):
    path = tmp_path / "m.pkl"
    joblib.dump({"a": 1}, path)
    assert joblib.load(path) == {"a": 1}
    model = MOD.load_model(str(path))
    assert isinstance(model, dict)


def test_load_model_missing(tmp_path):
    path = tmp_path / "missing.pkl"
    assert MOD.load_model(str(path)) is None


def test_load_model_ensemble(tmp_path):
    paths = [tmp_path / f"m{i}.pkl" for i in range(3)]
    for p in paths:
        joblib.dump({"p": p.name}, p)
    MOD.MODEL_RF_PATH = str(paths[0])
    MOD.MODEL_XGB_PATH = str(paths[1])
    MOD.MODEL_LGB_PATH = str(paths[2])
    model = MOD.load_model(str(paths[0]))
    assert hasattr(model, "models") and len(model.models) == 3


# --- health_check -----------------------------------------------------------
@pytest.mark.parametrize("rows,expected", [(0, False), (150, True)])
def test_health_check_various(monkeypatch, rows, expected):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "100")
    df = pd.DataFrame({"a": range(rows)}) if rows else pd.DataFrame()
    assert MOD.health_check(df, "d") is expected


# --- robustness -------------------------------------------------------------

def test_run_trading_cycle_empty_df_returns_no_orders():
    ctx = types.SimpleNamespace(api=None)
    df = pd.DataFrame({"price": []})
    assert MOD.run_trading_cycle(ctx, df) == []


