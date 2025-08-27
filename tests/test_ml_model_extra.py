from pathlib import Path

import numpy as np
import pytest
pd = pytest.importorskip("pandas")

from ai_trading import ml_model  # AI-AGENT-REF: canonical import
from ai_trading.ml_model import MLModel


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


def test_validate_errors():
    model = MLModel(DummyPipe())
    with pytest.raises(TypeError):
        model.predict([1, 2])
    df = make_df()
    df.loc[0, "a"] = np.nan
    with pytest.raises(ValueError):
        model.predict(df)


def test_fit_and_predict(tmp_path):
    model = MLModel(DummyPipe())
    df = make_df()
    mse = model.fit(df, np.array([0, 1]))
    assert mse >= 0
    preds = model.predict(df)
    assert len(preds) == len(df)
    save_path = model.save(tmp_path / "m.pkl")
    assert Path(save_path).exists()
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


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ml_model.load_model(str(tmp_path / "nonexistent.pkl"))


def test_save_and_load_model(tmp_path):
    dummy_model = {"foo": "bar"}
    model_path = tmp_path / "test_model.pkl"
    ml_model.save_model(dummy_model, str(model_path))
    loaded = ml_model.load_model(str(model_path))
    assert loaded == dummy_model
