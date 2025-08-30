from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

from ai_trading import ml_model  # AI-AGENT-REF: canonical import


def test_save_and_load_roundtrip():
    if ml_model.joblib is None:
        pytest.skip("joblib not available")
    model_dir = Path(ml_model.__file__).resolve().parent / "models"
    pipe = types.SimpleNamespace(
        fit=lambda X, y: None, predict=lambda X: [0] * len(X), version="1"
    )
    model = ml_model.MLModel(pipe)
    df = pd.DataFrame({"a": [1.0]})
    model.fit(df, [1.0])
    path = model_dir / "roundtrip.pkl"
    saved = model.save(path)
    loaded = ml_model.MLModel.load(saved)
    assert isinstance(loaded.pipeline, types.SimpleNamespace)
    Path(saved).unlink()


def test_save_outside_model_dir_raises(tmp_path):
    pipe = types.SimpleNamespace(fit=lambda X, y: None, predict=lambda X: [0])
    model = ml_model.MLModel(pipe)
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "m.pkl")

