from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

from ai_trading import ml_model  # AI-AGENT-REF: canonical import


def _fit_noop(X, y):
    return None


def _predict_zero(X):
    return [0] * len(X)


def _make_pipeline() -> types.SimpleNamespace:
    return types.SimpleNamespace(fit=_fit_noop, predict=_predict_zero, version="1")


def test_save_and_load_roundtrip():
    if ml_model.joblib is None:
        pytest.skip("joblib not available")
    model_dir = Path(ml_model.__file__).resolve().parent / "models"
    pipe = _make_pipeline()
    model = ml_model.MLModel(pipe)
    df = pd.DataFrame({"a": [1.0]})
    model.fit(df, [1.0])
    path = model_dir / "roundtrip.pkl"
    saved = model.save(path)
    loaded = ml_model.MLModel.load(saved)
    assert isinstance(loaded.pipeline, types.SimpleNamespace)
    Path(saved).unlink()


def test_save_outside_model_dir_raises(tmp_path):
    pipe = _make_pipeline()
    model = ml_model.MLModel(pipe)
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "m.pkl")

