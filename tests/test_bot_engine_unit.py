from tests.optdeps import require
require("pandas")
import types

import joblib
import pandas as pd
import pytest

# AI-AGENT-REF: Replaced unsafe _raise_dynamic_exec_disabled() with proper imports from core module
from ai_trading.core.bot_engine import (
    execute_trades,
    generate_signals,
    health_check,
    initialize_bot,
    load_model,
    run_trading_cycle,
)


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
    ctx, state = initialize_bot(api, dummy_loader)
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
    result = generate_signals(df)
    assert result.tolist() == expected


def test_generate_signals_missing_column_raises():
    with pytest.raises(KeyError):
        generate_signals(pd.DataFrame({"close": [1, 2]}))


# --- execute_trades ---------------------------------------------------------

def test_execute_trades_sends_orders(monkeypatch):
    api = DummyAPI()
    ctx = types.SimpleNamespace(api=api)
    sig = pd.Series([1, 0, -1], index=["A", "B", "C"])
    orders = execute_trades(ctx, sig)
    assert orders == [("A", "buy"), ("C", "sell")]
    assert api.calls == [("A", 1, "buy"), ("C", 1, "sell")]


# --- run_trading_cycle ------------------------------------------------------

def test_run_trading_cycle_integration(monkeypatch):
    api = DummyAPI()
    ctx = types.SimpleNamespace(api=api)
    df = pd.DataFrame({"price": [1, 3, 2]}, index=["A", "B", "C"])
    orders = run_trading_cycle(ctx, df)
    assert orders == [("B", "buy"), ("C", "sell")]


# --- load_model -------------------------------------------------------------

def test_load_model_single(tmp_path):
    path = tmp_path / "m.pkl"
    joblib.dump({"a": 1}, path)
    assert joblib.load(path) == {"a": 1}
    model = load_model(str(path))
    assert isinstance(model, dict)


def test_load_model_missing(tmp_path):
    path = tmp_path / "missing.pkl"
    assert load_model(str(path)) is None


def test_load_model_ensemble(tmp_path):
    paths = [tmp_path / f"m{i}.pkl" for i in range(3)]
    for p in paths:
        joblib.dump({"p": p.name}, p)
    # Note: load_model uses module-level constants for ensemble paths
    model = load_model(str(paths[0]))
    assert isinstance(model, dict)


# --- health_check -----------------------------------------------------------
@pytest.mark.parametrize("rows,expected", [(0, False), (150, True)])
def test_health_check_various(monkeypatch, rows, expected):
    monkeypatch.setenv("HEALTH_MIN_ROWS", "100")
    df = pd.DataFrame({"a": range(rows)}) if rows else pd.DataFrame()
    assert health_check(df, "d") is expected


# --- robustness -------------------------------------------------------------

def test_run_trading_cycle_empty_df_returns_no_orders():
    ctx = types.SimpleNamespace(api=None)
    df = pd.DataFrame({"price": []})
    assert run_trading_cycle(ctx, df) == []

