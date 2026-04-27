from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

import ai_trading.core.bot_engine as bot_engine
import ai_trading.model_loader as model_loader
from ai_trading.models.artifacts import write_artifact_manifest
from sklearn.dummy import DummyClassifier
import joblib


@pytest.fixture(autouse=True)
def isolated_model_loader_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_TRADING_MODELS_DIR", str(tmp_path / "models"))
    import ai_trading.paths as paths

    importlib.reload(paths)
    importlib.reload(model_loader)
    model_loader.ML_MODELS.clear()
    bot_engine._ML_MODEL_CACHE.clear()
    yield
    model_loader.ML_MODELS.clear()
    bot_engine._ML_MODEL_CACHE.clear()


def test_load_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_TRADING_MODELS_DIR", str(tmp_path / "ext"))
    import ai_trading.paths as paths

    importlib.reload(paths)
    ml = importlib.reload(model_loader)
    monkeypatch.setattr(ml, "INTERNAL_MODELS_DIR", tmp_path / "int")

    with pytest.raises(RuntimeError):
        ml.load_model("MISSING")


def test_load_corrupt_logs_error(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("AI_TRADING_MODELS_DIR", str(tmp_path))
    import ai_trading.paths as paths

    importlib.reload(paths)
    ml = importlib.reload(model_loader)
    be = importlib.reload(bot_engine)
    ml.ML_MODELS.clear()
    be._ML_MODEL_CACHE.clear()
    monkeypatch.setattr(ml, "INTERNAL_MODELS_DIR", tmp_path)

    bad = tmp_path / "BAD.pkl"
    bad.write_text("not a model")
    caplog.set_level("ERROR")
    result = be._load_ml_model("BAD")
    assert result is None
    assert any(r.message.startswith("MODEL_LOAD_ERROR") for r in caplog.records)


def test_load_real_model():
    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model.fit(X, y)
    model_loader.ML_MODELS["TESTSYM"] = model
    loaded = bot_engine._load_ml_model("TESTSYM")
    assert loaded is not None
    pred = loaded.predict([[0]])[0]
    assert pred in [0, 1]


def test_load_ml_model_uses_cached_registry(monkeypatch):
    class DummyModel:
        def predict(self, X):  # pragma: no cover - simple stub
            return X

        def predict_proba(self, X):  # pragma: no cover - simple stub
            return X

    cached_model = DummyModel()

    def fail_load(symbol: str):  # pragma: no cover - should not be invoked
        raise AssertionError("load_model should not be called when cache is primed")

    monkeypatch.setattr(model_loader, "load_model", fail_load)
    model_loader.ML_MODELS["CACHE"] = cached_model

    loaded = bot_engine._load_ml_model("CACHE")

    assert loaded is cached_model
    assert bot_engine._ML_MODEL_CACHE["CACHE"] is cached_model
    assert model_loader.ML_MODELS["CACHE"] is cached_model


def test_signal_ml_returns_prediction_and_probability():
    pd = pytest.importorskip("pandas")
    model = DummyClassifier(strategy="prior")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model.fit(X, y)
    df = pd.DataFrame(
        {
            "rsi": [50],
            "macd": [1.0],
            "atr": [1.0],
            "vwap": [1.0],
            "sma_50": [1.0],
            "sma_200": [1.0],
        }
    )
    sm = bot_engine.SignalManager()
    result = sm.signal_ml(df, model=model)
    assert result is not None
    signal, proba, label = result
    assert label == "ml"
    assert signal in (-1, 1)
    assert 0.0 <= proba <= 1.0


def test_load_model_from_external_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_TRADING_MODELS_DIR", str(tmp_path))
    import ai_trading.paths as paths

    importlib.reload(paths)
    ml = importlib.reload(model_loader)

    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model.fit(X, y)
    joblib.dump(model, tmp_path / "EXT.pkl")

    loaded = ml.load_model("EXT")
    assert hasattr(loaded, "predict")


def test_load_model_from_internal_dir(tmp_path, monkeypatch):
    external = tmp_path / "ext"
    internal = tmp_path / "int"
    external.mkdir()
    internal.mkdir()
    monkeypatch.setenv("AI_TRADING_MODELS_DIR", str(external))
    import ai_trading.paths as paths

    importlib.reload(paths)
    ml = importlib.reload(model_loader)
    monkeypatch.setattr(ml, "INTERNAL_MODELS_DIR", internal)

    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model.fit(X, y)
    joblib.dump(model, internal / "INT.pkl")

    loaded = ml.load_model("INT")
    assert hasattr(loaded, "predict")


def test_live_model_loader_requires_verified_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_TRADING_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("AI_TRADING_MODEL_VERIFY_CHECKSUM", "1")
    import ai_trading.paths as paths

    importlib.reload(paths)
    ml = importlib.reload(model_loader)

    model = DummyClassifier(strategy="most_frequent")
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    model.fit(X, y)
    model_path = tmp_path / "LIVE.pkl"
    joblib.dump(model, model_path)

    with pytest.raises(RuntimeError, match="MODEL_VERIFICATION_FAILED"):
        ml.load_model("LIVE")

    write_artifact_manifest(model_path=model_path, model_version="live-test-v1")

    loaded = ml.load_model("LIVE")
    assert hasattr(loaded, "predict")
