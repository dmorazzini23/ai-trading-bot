from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from ai_trading.core import bot_engine as bot
from ai_trading.core.bot_engine import _ML_MODEL_CACHE, _load_ml_model
from ai_trading.model_loader import ML_MODELS
from sklearn.dummy import DummyClassifier


def setup_function(function):
    ML_MODELS.clear()
    _ML_MODEL_CACHE.clear()
    from pathlib import Path

    models_dir = Path(__file__).resolve().parents[1] / "models"
    for p in models_dir.rglob("*.pkl"):
        try:
            p.unlink()
        except OSError:
            pass
    for d in sorted(models_dir.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass


def test_load_missing_trains_model(caplog):
    caplog.set_level("WARNING")
    result = _load_ml_model("FAKE")
    assert result is not None
    assert hasattr(result, "predict_proba")
    from pathlib import Path

    assert (Path("models") / "FAKE.pkl").exists()
    assert any(r.message.startswith("MODEL_FILE_MISSING") for r in caplog.records)


def test_load_corrupt_logs_error(tmp_path, monkeypatch, caplog):
    from pathlib import Path

    monkeypatch.setenv("MODELS_DIR", str(Path("models") / "tmp"))
    bad = Path("models") / "tmp" / "BAD.pkl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not a model")
    caplog.set_level("ERROR")
    result = _load_ml_model("BAD")
    assert result is None
    assert any(r.message.startswith("MODEL_LOAD_ERROR") for r in caplog.records)


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
    sm = bot.SignalManager()
    result = sm.signal_ml(df, model=model)
    assert result is not None
    signal, proba, label = result
    assert label == "ml"
    assert signal in (-1, 1)
    assert 0.0 <= proba <= 1.0

