from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
pd = pytest.importorskip("pandas")

from ai_trading import ml_model  # AI-AGENT-REF: canonical import
from ai_trading.models.artifacts import default_manifest_path


class DummyPipe:
    def __init__(self):
        self.fitted = False
        self.version = "1"

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("not fitted")
        return np.ones(len(X))


def make_df():
    return pd.DataFrame({"a": [1.0, 2.0]})


@pytest.fixture
def isolated_ml_model_dir(tmp_path, monkeypatch):
    model_root = tmp_path / "pkg" / "models"
    model_root.mkdir(parents=True)
    monkeypatch.setattr(ml_model, "__file__", str(tmp_path / "pkg" / "ml_model.py"))
    return model_root






def test_train_model_invalid_algorithm():
    with pytest.raises(ValueError):
        ml_model.train_model([], [], algorithm="bad_algo")


def test_train_model_invalid_data():
    with pytest.raises(ValueError):
        ml_model.train_model(None, None)






def test_load_model_missing_file(isolated_ml_model_dir):
    with pytest.raises(FileNotFoundError):
        ml_model.load_model(str(isolated_ml_model_dir / "nonexistent.pkl"))


def test_save_and_load_model(isolated_ml_model_dir):
    if ml_model.joblib is None:
        pytest.skip("joblib not available")
    dummy_model = {"foo": "bar"}
    model_path = isolated_ml_model_dir / "test_model.pkl"
    ml_model.save_model(dummy_model, str(model_path))
    assert default_manifest_path(model_path).exists()
    loaded = ml_model.load_model(str(model_path))
    assert loaded == dummy_model




