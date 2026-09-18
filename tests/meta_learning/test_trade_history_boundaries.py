"""Trade-history adapters must preserve observations and handle absent inputs."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def history(monkeypatch, tmp_path):
    from ai_trading.meta_learning import core
    path = tmp_path / "observed.csv"
    monkeypatch.setattr(core, "pd", pd)
    monkeypatch.setattr(core, "config", SimpleNamespace(
        TRADE_LOG_FILE=str(path), META_LEARNING_LOG_FILE=str(tmp_path / "meta.csv")))
    monkeypatch.delitem(sys.modules, "bot_engine", raising=False)
    monkeypatch.delitem(sys.modules, "config", raising=False)
    return core, path


def test_signal_performance_uses_observed_direction_and_minimum_support(history):
    core, path = history
    pd.DataFrame([
        {"entry_price": 100, "exit_price": 102, "signal_tags": "trend+shared", "side": "buy"},
        {"entry_price": 100, "exit_price": 98, "signal_tags": "trend+shared", "side": "sell"},
        {"entry_price": 100, "exit_price": 101, "signal_tags": "shared", "side": "sell"},
        {"entry_price": "invalid", "exit_price": 100, "signal_tags": "bad", "side": "buy"},
        {"entry_price": 100, "exit_price": None, "signal_tags": "unclosed", "side": "buy"},
    ]).to_csv(path, index=False)
    assert core.load_global_signal_performance(min_trades=2, threshold=0.4) == {"trend": 2.0, "shared": 1.0}
    assert core.load_global_signal_performance(min_trades=3, threshold=1.5) == {}


@pytest.mark.parametrize("content", [None, "wrong,columns\n1,2\n",
    "entry_price,exit_price,signal_tags,side\n",
    "entry_price,exit_price,signal_tags,side\nbad,no,trend,buy\n"])
def test_missing_or_unusable_trade_history_returns_no_signal_performance(history, content):
    core, path = history
    if content is not None:
        path.write_text(content)
    assert core.load_global_signal_performance() == {}


def test_signal_performance_delegation_requires_mapping(history, monkeypatch):
    core, _ = history
    monkeypatch.setitem(sys.modules, "bot_engine", SimpleNamespace(load_global_signal_performance=lambda *a: {"trend": 2}))
    assert core.load_global_signal_performance() == {"trend": 2}
    monkeypatch.setitem(sys.modules, "bot_engine", SimpleNamespace(load_global_signal_performance=lambda *a: None))
    assert core.load_global_signal_performance() == {}
    monkeypatch.setitem(sys.modules, "bot_engine", SimpleNamespace())
    assert core.load_global_signal_performance() == {}


def test_observed_round_trip_conversion_keeps_prices_and_timestamps(history):
    core, _ = history
    rows = [
        ["00000000-0000-0000-0000-000000000001", "2026-08-03T14:00Z", "AAPL", "buy", 2, 100, "paper", "filled"],
        ["00000000-0000-0000-0000-000000000002", "2026-08-03T15:00Z", "AAPL", "sell", 2, 102, "paper", "filled"],
        ["00000000-0000-0000-0000-000000000003", "2026-08-03T15:00Z", "MSFT", "buy", 1, 200, "paper", "canceled"],
    ]
    result = core._convert_mixed_format_to_meta(pd.DataFrame(rows))
    assert len(result) == 1
    row = result.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["entry_price"] == 100 and row["exit_price"] == 102
    assert row["qty"] == 2
    assert row["entry_time"] == rows[0][1] and row["exit_time"] == rows[1][1]
    assert row["reward"] == pytest.approx(0.02)


def test_stored_meta_history_appends_without_repeating_headers(history):
    core, _ = history
    first = {"symbol": "AAPL", "entry_price": 100, "exit_price": 102}
    second = {"symbol": "MSFT", "entry_price": 200, "exit_price": 198}
    assert core.store_meta_learning_data(first)
    assert core.store_meta_learning_data(second)
    result = pd.read_csv(core.config.META_LEARNING_LOG_FILE)
    assert result.to_dict("records") == [first, second]


@pytest.mark.parametrize("numpy_available", [True, False])
def test_loaded_optimizer_bounds_predictions_and_scales_for_volatility(history, monkeypatch, numpy_available):
    core, _ = history
    monkeypatch.setattr(core, "load_model_checkpoint", lambda path: SimpleNamespace(predict=lambda data: [-2.0, 0.4, 2.0]))
    monkeypatch.setattr(core, "_import_numpy", lambda **kw: np if numpy_available else None)
    result = core.optimize_signals([1, 2, 3], SimpleNamespace(MODEL_PATH="unused"), volatility=2)
    assert result == pytest.approx([-0.6, 0.2, 0.6])


@pytest.mark.parametrize("prediction", [None, ValueError("bad features")])
def test_prediction_failure_preserves_input_without_inventing_signals(history, monkeypatch, prediction):
    core, _ = history
    def predict(data):
        if isinstance(prediction, Exception):
            raise prediction
        return prediction
    model = SimpleNamespace(predict=predict)
    monkeypatch.setattr(core, "load_model_checkpoint", lambda path: model)
    source = [0.3, -0.1]
    assert core.optimize_signals(source, model=model) == source
    assert core.optimize_signals(source, SimpleNamespace(MODEL_PATH="unused")) == source


@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_conversion_trigger_rejects_unavailable_files(history, kind):
    core, path = history
    if kind == "empty":
        path.touch()
    elif kind == "directory":
        path.mkdir()
    assert not core.trigger_meta_learning_conversion({"symbol": "AAPL"})
