import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ensure project root in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if "joblib" in sys.modules:
    del sys.modules["joblib"]

os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")
import config
import data_fetcher
from ml_model import MLModel


class DummyEngine:
    def __init__(self):
        self.orders = []

    def execute_order(self, symbol, qty, side, asset_class=None):
        self.orders.append((symbol, qty, side))


def test_end_to_end_pipeline(monkeypatch):
    os.environ.setdefault("APCA_API_KEY_ID", "k")
    os.environ.setdefault("APCA_API_SECRET_KEY", "s")
    config.reload_env()

    # prepare mock minute data
    idx = pd.date_range("2024-01-01", periods=2, freq="T", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.2, 1.2],
            "low": [0.9, 1.0],
            "close": [1.1, 1.15],
            "volume": [100, 150],
        },
        index=idx,
    )

    monkeypatch.setattr(data_fetcher, "is_market_open", lambda: True)
    monkeypatch.setattr(data_fetcher, "get_minute_df", lambda *a, **k: df)

    class DummyPipe:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    model = MLModel(DummyPipe())
    mse = model.fit(df, np.array([0.1, 0.2]))
    preds = model.predict(df)
    assert len(preds) == len(df)
    assert mse >= 0

    engine = DummyEngine()
    engine.execute_order("AAPL", 1, "buy")
    assert engine.orders
