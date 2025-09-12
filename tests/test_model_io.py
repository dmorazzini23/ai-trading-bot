from pathlib import Path

import pytest

dill = pytest.importorskip("dill")

from ai_trading.ml import model_io


def test_save_and_load_lambda(tmp_path):
    model = {"f": lambda x: x + 1}
    path = tmp_path / "lambda.pkl"
    model_io.save_model(model, path)
    loaded = model_io.load_model(path)
    assert loaded["f"](1) == 2


def test_load_missing_model_raises(tmp_path):
    missing = tmp_path / "missing.pkl"
    with pytest.raises(RuntimeError, match="Model file not found"):
        model_io.load_model(missing)
