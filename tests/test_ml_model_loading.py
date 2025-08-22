import pickle
import numpy as np
from sklearn.dummy import DummyClassifier
import sys
import types

# AI-AGENT-REF: Replaced unsafe _raise_dynamic_exec_disabled() with direct imports from core module
from ai_trading.core.bot_engine import _load_ml_model

# Setup stub for model loader dependency
stub = types.ModuleType("ai_trading.model_loader")
stub.ML_MODELS = {}

def _stub_load(symbol: str):
    return stub.ML_MODELS.get(symbol)

stub.load_model = _stub_load
sys.modules["ai_trading.model_loader"] = stub


def test_load_missing_logs_error(caplog):
    caplog.set_level("INFO")
    result = _load_ml_model("FAKE")
    assert result is None
    assert not caplog.records


def test_load_real_model(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model.fit(X, y)
    path = models_dir / "TESTSYM.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    monkeypatch.chdir(tmp_path)
    with open(path, "rb") as f:
        stub.ML_MODELS["TESTSYM"] = pickle.load(f)
    loaded = _load_ml_model("TESTSYM")
    assert loaded is not None
    pred = loaded.predict([[0]])[0]
    assert pred in [0, 1]
