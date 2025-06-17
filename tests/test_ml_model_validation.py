import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_model import MLModel


class DummyPipe:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_predict_validation(monkeypatch):
    model = MLModel(DummyPipe())
    df = pd.DataFrame({"a": [1.0, float("nan")]})
    try:
        model.predict(df)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"
