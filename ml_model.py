from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


class MLModel:
    """Minimal wrapper around an underlying pipeline."""
    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def fit(self, X: Iterable, y: Iterable):
        self.pipeline.fit(X, y)
        preds = self.pipeline.predict(X)
        return float(np.mean((np.asarray(preds) - np.asarray(y)) ** 2))

    def predict(self, X):
        if isinstance(X, list):
            raise TypeError("expects array-like input")
        return list(self.pipeline.predict(X))

    def save(self, path: Path | str) -> str:
        joblib.dump(self.pipeline, path)
        return str(path)

    @classmethod
    def load(cls, path: str):
        model = joblib.load(path)
        return cls(model)


def train_model(X, y, algorithm: str = "linear"):
    if X is None or y is None:
        raise ValueError("training data required")
    if algorithm != "linear":
        raise ValueError("unsupported algorithm")
    model = LinearRegression().fit(X, y)
    return model


def predict_model(model, X):
    if model is None or X is None:
        raise ValueError("model and input required")
    if not X:
        return []
    return list(model.predict(X))


def load_model(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)
    return joblib.load(path_obj)


def save_model(model, path: str):
    joblib.dump(model, path)
    return path
