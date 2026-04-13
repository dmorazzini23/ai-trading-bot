import importlib

import pandas as pd


def reload_bot_engine():
    return importlib.reload(__import__('ai_trading.core.bot_engine', fromlist=['dummy']))


def test_signal_ml_executes_prediction(monkeypatch):
    be = reload_bot_engine()

    class DummyModel:
        def predict(self, X):  # noqa: D401
            """Record prediction call."""
            return [1]

        def predict_proba(self, X):  # noqa: D401
            """Return constant probability."""
            return [[0.2, 0.8]]

    dummy = DummyModel()
    monkeypatch.setattr(be, "_load_ml_model", lambda symbol: dummy)
    sm = be.SignalManager()
    df = pd.DataFrame(
        {
            "rsi": [50],
            "macd": [0],
            "atr": [1],
            "vwap": [100],
            "sma_50": [100],
            "sma_200": [100],
        }
    )
    result = sm.signal_ml(df, symbol="SPY")
    assert result == (1, 0.8, "ml")


def test_signal_ml_applies_inverse_score_orientation(monkeypatch):
    be = reload_bot_engine()

    class DummyModel:
        classes_ = [0, 1]
        edge_score_orientation_ = "inverse"

        def predict(self, X):  # noqa: D401
            return [0]

        def predict_proba(self, X):  # noqa: D401
            return [[0.8, 0.2]]

    dummy = DummyModel()
    monkeypatch.setenv("AI_TRADING_ML_USE_REGIME_THRESHOLDS", "0")
    monkeypatch.setattr(be, "_load_ml_model", lambda symbol: dummy)
    sm = be.SignalManager()
    df = pd.DataFrame(
        {
            "rsi": [50],
            "macd": [0],
            "atr": [1],
            "vwap": [100],
            "sma_50": [100],
            "sma_200": [100],
        }
    )
    result = sm.signal_ml(df, symbol="SPY")
    assert result == (1, 0.8, "ml")


def test_signal_ml_symbol_penalty_can_block_trade(monkeypatch):
    be = reload_bot_engine()

    class DummyModel:
        classes_ = [0, 1]
        edge_negative_symbol_penalties_ = {
            "SPY": {"threshold_bump": 0.55, "confidence_scale": 0.7}
        }

        def __init__(self):
            self.called = False

        def predict(self, X):  # noqa: D401
            self.called = True
            return [1]

        def predict_proba(self, X):  # noqa: D401
            return [[0.4, 0.6]]

    dummy = DummyModel()
    monkeypatch.setenv("AI_TRADING_ML_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("AI_TRADING_ML_USE_REGIME_THRESHOLDS", "0")
    monkeypatch.setattr(be, "_load_ml_model", lambda symbol: dummy)
    sm = be.SignalManager()
    df = pd.DataFrame(
        {
            "rsi": [50],
            "macd": [0],
            "atr": [1],
            "vwap": [100],
            "sma_50": [100],
            "sma_200": [100],
        }
    )
    assert sm.signal_ml(df, symbol="SPY") is None
