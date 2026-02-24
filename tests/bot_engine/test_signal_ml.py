import sys

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core.bot_engine import SignalManager


def _minimal_df() -> pd.DataFrame:
    """Return a DataFrame with required features for ML and VSA heuristics."""
    data = {
        "rsi": [30.0] * 20,
        "macd": [0.1] * 20,
        "atr": [1.0] * 20,
        "vwap": [1.0] * 20,
        "sma_50": [1.0] * 20,
        "sma_200": [1.0] * 20,
        "close": [1.0] * 20,
        "open": [1.0] * 20,
        "volume": [1.0] * 20,
    }
    return pd.DataFrame(data)


def test_signal_ml_with_dummy_model(monkeypatch, caplog):
    """Calling signal_ml with a dummy model should emit no warnings."""
    monkeypatch.setenv("AI_TRADING_MODEL_MODULE", "dummy_model")

    dummy_model = sys.modules["dummy_model"].get_model()
    manager = SignalManager()
    df = _minimal_df()

    caplog.set_level("WARNING")
    result = manager.signal_ml(df, model=dummy_model)

    assert result is not None
    assert "ML predictions disabled" not in caplog.text


def test_signal_ml_warns_once(monkeypatch, caplog):
    """Missing models trigger a single warning and disable ML signals by default."""
    monkeypatch.delenv("AI_TRADING_MODEL_MODULE", raising=False)
    monkeypatch.delenv("AI_TRADING_MODEL_PATH", raising=False)
    manager = SignalManager()
    df = _minimal_df()

    caplog.set_level("WARNING")
    first = manager.signal_ml(df)
    second = manager.signal_ml(df)

    warnings = [
        rec for rec in caplog.records if "ML predictions disabled" in rec.getMessage()
    ]
    assert first is None
    assert second is None
    assert len(warnings) == 1


def test_signal_ml_returns_none_when_features_missing(caplog):
    class DummyModel:
        feature_names_in_ = ["feature_x"]

        def predict(self, _X):
            return [1]

        def predict_proba(self, _X):
            return [[0.2, 0.8]]

    manager = SignalManager()
    df = _minimal_df()
    caplog.set_level("ERROR")

    result = manager.signal_ml(df, model=DummyModel(), symbol="AAPL")

    assert result is None
    assert "ML_SIGNAL_MISSING_FEATURES" in caplog.text


def test_signal_ml_prediction_error_returns_none(caplog):
    class DummyModel:
        feature_names_in_ = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]

        def predict(self, _X):
            raise ValueError("bad input")

        def predict_proba(self, _X):
            return [[0.5, 0.5]]

    manager = SignalManager()
    df = _minimal_df()
    caplog.set_level("ERROR")

    result = manager.signal_ml(df, model=DummyModel(), symbol="AAPL")

    assert result is None
    assert "signal_ml predict failed" in caplog.text


def test_signal_ml_respects_min_confidence(monkeypatch, caplog):
    class DummyModel:
        feature_names_in_ = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]

        def predict(self, _X):
            return [1]

        def predict_proba(self, _X):
            return [[0.2, 0.8]]

    monkeypatch.setenv("AI_TRADING_ML_MIN_CONFIDENCE", "0.95")
    manager = SignalManager()
    df = _minimal_df()
    caplog.set_level("INFO")

    result = manager.signal_ml(df, model=DummyModel(), symbol="AAPL")

    assert result is None
    assert "ML_SIGNAL_BELOW_CONFIDENCE" in caplog.text


def test_signal_ml_respects_regime_thresholds(monkeypatch, caplog):
    class DummyModel:
        feature_names_in_ = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]
        edge_thresholds_by_regime_ = {"uptrend": 0.9, "sideways": 0.4}
        edge_global_threshold_ = 0.5

        def predict(self, _X):
            return [1]

        def predict_proba(self, _X):
            return [[0.2, 0.8]]

    monkeypatch.setenv("AI_TRADING_ML_MIN_CONFIDENCE", "0.0")
    monkeypatch.setenv("AI_TRADING_ML_USE_REGIME_THRESHOLDS", "1")
    manager = SignalManager()
    df = pd.concat([_minimal_df(), _minimal_df()], ignore_index=True)
    df["close"] = [100.0 + idx for idx in range(len(df))]
    caplog.set_level("INFO")

    result = manager.signal_ml(df, model=DummyModel(), symbol="AAPL")

    assert result is None
    assert "ML_SIGNAL_BELOW_CONFIDENCE" in caplog.text


def test_signal_ml_can_disable_regime_thresholds(monkeypatch):
    class DummyModel:
        feature_names_in_ = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]
        edge_thresholds_by_regime_ = {"uptrend": 0.9, "sideways": 0.4}
        edge_global_threshold_ = 0.5

        def predict(self, _X):
            return [1]

        def predict_proba(self, _X):
            return [[0.2, 0.8]]

    monkeypatch.setenv("AI_TRADING_ML_MIN_CONFIDENCE", "0.0")
    monkeypatch.setenv("AI_TRADING_ML_USE_REGIME_THRESHOLDS", "0")
    manager = SignalManager()
    df = pd.concat([_minimal_df(), _minimal_df()], ignore_index=True)
    df["close"] = [100.0 + idx for idx in range(len(df))]

    result = manager.signal_ml(df, model=DummyModel(), symbol="AAPL")

    assert result is not None
    signal, confidence, label = result
    assert signal == 1
    assert confidence == pytest.approx(0.8, abs=1e-6)
    assert label == "ml"
