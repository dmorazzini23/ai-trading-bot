from __future__ import annotations

import importlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

import ai_trading.core.bot_engine as bot_engine
import ai_trading.model_loader as model_loader
from ai_trading.models.artifacts import default_manifest_path, write_artifact_manifest
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


def test_train_and_save_model_synthetic_fallback_is_test_only_and_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr("ai_trading.data.fetch.get_daily_df", lambda *_args, **_kwargs: None)

    model = model_loader.train_and_save_model("SYN", tmp_path)

    model_path = tmp_path / "SYN.pkl"
    assert hasattr(model, "predict")
    assert not model_path.exists()
    assert not default_manifest_path(model_path).exists()
    assert set(model.predict(np.zeros((4, 9))).tolist()) <= {0, 1}


def test_train_and_save_model_fails_closed_without_real_bars_in_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("ai_trading.data.fetch.get_daily_df", lambda *_args, **_kwargs: None)

    def fake_get_env(key, default=None, **_kwargs):
        if key in {
            "PYTEST_CURRENT_TEST",
            "PYTEST_RUNNING",
            "TESTING",
            "AI_TRADING_MODEL_TRAINING_SMOKE",
        }:
            return "" if isinstance(default, str) else False
        return default

    monkeypatch.setattr(model_loader, "get_env", fake_get_env)

    with pytest.raises(RuntimeError, match="Real training bars unavailable"):
        model_loader.train_and_save_model("SYN", tmp_path)


def test_training_frame_trend_uses_past_only_for_future_tail_mutation():
    pd = pytest.importorskip("pandas")
    rows = 90
    close = 100.0 + np.arange(rows, dtype=float) * 0.1 + np.sin(np.arange(rows) / 4.0)
    base = pd.DataFrame(
        {"close": close, "volume": np.linspace(100_000.0, 120_000.0, rows)},
        index=pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC"),
    )
    mutated = base.copy()
    mutation_start = base.index[70]
    mutated.loc[mutation_start:, "close"] = mutated.loc[mutation_start:, "close"] + 10_000.0

    base_features = model_loader._build_training_frame(base)
    mutated_features = model_loader._build_training_frame(mutated)

    past_index = base_features.index[base_features.index < mutation_start]
    pd.testing.assert_series_equal(
        base_features.loc[past_index, "trend"],
        mutated_features.loc[past_index, "trend"],
    )


def test_training_frame_drops_final_missing_future_close_before_labeling():
    pd = pytest.importorskip("pandas")
    rows = 40
    frame = pd.DataFrame(
        {"close": np.arange(100.0, 100.0 + rows), "volume": np.linspace(100.0, 200.0, rows)},
        index=pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC"),
    )

    features = model_loader._build_training_frame(frame)

    assert frame.index[-1] not in features.index
    assert set(features["y"].astype(int).unique()) == {1}


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
