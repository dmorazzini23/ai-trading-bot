import importlib

import pandas as pd


def reload_bot_engine():
    return importlib.reload(__import__('ai_trading.core.bot_engine', fromlist=['dummy']))


def test_signal_ml_executes_prediction(monkeypatch):
    be = reload_bot_engine()

    class DummyModel:
        def __init__(self):
            self.called = False

        def predict(self, X):  # noqa: D401
            """Record prediction call."""
            self.called = True
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
    assert dummy.called
    assert result == (1, 0.8, "ml")
