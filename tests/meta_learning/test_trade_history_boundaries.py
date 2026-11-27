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

