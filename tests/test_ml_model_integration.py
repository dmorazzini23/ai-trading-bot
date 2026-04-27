from __future__ import annotations

import types

import pandas as pd
import pytest

from ai_trading import ml_model  # AI-AGENT-REF: canonical import
from ai_trading.models.artifacts import default_manifest_path


def _fit_noop(X, y):
    return None


def _predict_zero(X):
    return [0] * len(X)


def _make_pipeline() -> types.SimpleNamespace:
    return types.SimpleNamespace(fit=_fit_noop, predict=_predict_zero, version="1")


@pytest.fixture
def isolated_ml_model_dir(tmp_path, monkeypatch):
    model_root = tmp_path / "pkg" / "models"
    model_root.mkdir(parents=True)
    monkeypatch.setattr(ml_model, "__file__", str(tmp_path / "pkg" / "ml_model.py"))
    return model_root


def test_save_and_load_roundtrip(isolated_ml_model_dir):
    if ml_model.joblib is None:
        pytest.skip("joblib not available")
    pipe = _make_pipeline()
    model = ml_model.MLModel(pipe)
    df = pd.DataFrame({"a": [1.0]})
    model.fit(df, [1.0])
    path = isolated_ml_model_dir / "roundtrip.pkl"
    saved = model.save(str(path))
    assert default_manifest_path(saved).exists()
    loaded = ml_model.MLModel.load(saved)
    assert isinstance(loaded.pipeline, types.SimpleNamespace)


def test_save_outside_model_dir_raises(tmp_path):
    pipe = _make_pipeline()
    model = ml_model.MLModel(pipe)
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "m.pkl")
