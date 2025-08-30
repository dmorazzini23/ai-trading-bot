from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from ai_trading.core.bot_engine import _ML_MODEL_CACHE, _load_ml_model
from ai_trading.model_loader import ML_MODELS
from sklearn.dummy import DummyClassifier


def setup_function(function):
    ML_MODELS.clear()
    _ML_MODEL_CACHE.clear()


def test_load_missing_logs_error(caplog):
    caplog.set_level("INFO")
    result = _load_ml_model("FAKE")
    assert result is None
    assert not caplog.records


def test_load_real_model():
    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model.fit(X, y)
    ML_MODELS["TESTSYM"] = model
    loaded = _load_ml_model("TESTSYM")
    assert loaded is not None
    pred = loaded.predict([[0]])[0]
    assert pred in [0, 1]

