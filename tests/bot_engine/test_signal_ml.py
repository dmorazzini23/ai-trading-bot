import json
import sys
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.core import bot_engine
from ai_trading.core.bot_engine import SignalManager


def _minimal_df() -> Any:
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


def test_signal_ml_derives_after_hours_features_when_missing() -> None:
    class DummyModel:
        feature_names_in_ = [
            "rsi",
            "macd",
            "atr",
            "vwap",
            "sma_50",
            "sma_200",
            "signal",
            "atr_pct",
            "vwap_distance",
            "sma_spread",
            "macd_signal_gap",
            "rsi_centered",
        ]

        def predict(self, _X):
            return [1]

        def predict_proba(self, _X):
            return [[0.2, 0.8]]

    manager = SignalManager()
    df = _minimal_df()

    result = manager.signal_ml(df, model=DummyModel(), symbol="AAPL")

    assert result is not None
    signal, confidence, label = result
    assert signal == 1
    assert confidence == pytest.approx(0.8, abs=1e-6)
    assert label == "ml"


def test_signal_ml_prediction_error_returns_none(caplog):
    class DummyModel:
        feature_names_in_ = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]

        def predict(self, _X):
            return [1]

        def predict_proba(self, _X):
            raise ValueError("bad input")

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


def test_signal_ml_shadow_logs_predictions(monkeypatch, tmp_path):
    class ChampionModel:
        feature_names_in_ = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]

        def predict(self, _X):
            return [1]

        def predict_proba(self, _X):
            return [[0.1, 0.9]]

    class ChallengerModel:
        feature_names_in_ = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]

        def predict(self, _X):
            return [0]

        def predict_proba(self, _X):
            return [[0.7, 0.3]]

    shadow_path = tmp_path / "ml_shadow.jsonl"
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SHADOW_LOG_PATH", str(shadow_path))
    monkeypatch.setattr(bot_engine, "_load_shadow_model", lambda: ChallengerModel())
    manager = SignalManager()

    result = manager.signal_ml(_minimal_df(), model=ChampionModel(), symbol="AAPL")

    assert result is not None
    payload = json.loads(shadow_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["symbol"] == "AAPL"
    assert payload["champion_probability"] == pytest.approx(0.9, abs=1e-6)
    assert payload["challenger_probability"] == pytest.approx(0.7, abs=1e-6)


def test_signal_ml_reports_training_serving_skew(monkeypatch, caplog):
    class SkewedModel:
        feature_names_in_ = ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]
        training_feature_stats_ = {
            "rsi": {"mean": 50.0, "std": 1.0, "p05": 49.0, "p95": 51.0},
            "macd": {"mean": 0.0, "std": 0.1, "p05": -0.1, "p95": 0.1},
            "atr": {"mean": 1.0, "std": 0.2, "p05": 0.7, "p95": 1.3},
            "vwap": {"mean": 1.0, "std": 0.05, "p05": 0.9, "p95": 1.1},
            "sma_50": {"mean": 1.0, "std": 0.05, "p05": 0.9, "p95": 1.1},
            "sma_200": {"mean": 1.0, "std": 0.05, "p05": 0.9, "p95": 1.1},
        }

        def predict(self, _X):
            return [1]

        def predict_proba(self, _X):
            return [[0.2, 0.8]]

    df = _minimal_df()
    df.loc[df.index[-1], "rsi"] = 90.0
    monkeypatch.setenv("AI_TRADING_ML_SKEW_MONITOR_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_ML_SKEW_MEAN_ABS_Z_THRESHOLD", "0.5")
    monkeypatch.setenv("AI_TRADING_ML_SKEW_OUTLIER_RATIO_THRESHOLD", "0.1")
    caplog.set_level("WARNING")
    manager = SignalManager()

    result = manager.signal_ml(df, model=SkewedModel(), symbol="AAPL")

    assert result is not None
    assert "ML_TRAINING_SERVING_SKEW" in caplog.text
