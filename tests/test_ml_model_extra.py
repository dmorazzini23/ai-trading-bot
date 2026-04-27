from pathlib import Path

import numpy as np
import pytest
pd = pytest.importorskip("pandas")

from ai_trading import ml_model  # AI-AGENT-REF: canonical import
from ai_trading.ml_model import MLModel
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


def test_validate_errors():
    model = MLModel(DummyPipe())
    with pytest.raises(TypeError):
        model.predict([1, 2])
    df = make_df()
    df.loc[0, "a"] = np.nan
    with pytest.raises(ValueError):
        model.predict(df)


def test_fit_and_predict(isolated_ml_model_dir):
    model = MLModel(DummyPipe())
    df = make_df()
    mse = model.fit(df, np.array([0, 1]))
    assert mse >= 0
    preds = model.predict(df)
    assert len(preds) == len(df)
    if ml_model.joblib is None:
        pytest.skip("joblib not available")
    save_path = model.save(str(isolated_ml_model_dir / "m.pkl"))
    assert Path(save_path).exists()
    assert default_manifest_path(save_path).exists()
    loaded = MLModel.load(save_path)
    assert isinstance(loaded.pipeline, DummyPipe)


def test_train_model_invalid_algorithm():
    with pytest.raises(ValueError):
        ml_model.train_model([], [], algorithm="bad_algo")


def test_train_model_invalid_data():
    with pytest.raises(ValueError):
        ml_model.train_model(None, None)


def test_predict_model_untrained():
    class DummyModel:
        def predict(self, X):
            raise AttributeError("not fitted")

    with pytest.raises(AttributeError):
        ml_model.predict_model(DummyModel(), [1, 2, 3])


def test_predict_model_invalid_input():
    class DummyModel:
        def predict(self, X):
            return [0] * len(X)

    result = ml_model.predict_model(DummyModel(), [])
    assert result == []


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
