from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class _DummyPipe:
    fitted: bool = False
    version: str = "1"

    def fit(self, X: Iterable, y: Iterable) -> _DummyPipe:
        self.fitted = True
        return self

    def predict(self, X: Iterable) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("not fitted")
        try:
            n = len(X)
        # noqa: BLE001 TODO: narrow exception
        except Exception:
            n = 0
        return np.zeros(n)


class MLModel:
    def __init__(self, model: Any | None = None):
        self.model = model or _DummyPipe()

    def _require(self, name: str) -> None:
        if not hasattr(self.model, name):
            raise TypeError(f"Model missing required method: {name}")

    def fit(self, X: Sequence, y: Sequence, sample_weight=None) -> MLModel:
        if not all(hasattr(self.model, m) for m in ("fit", "predict")):
            raise TypeError("Model missing required methods: fit, predict")
        import inspect

        sig = inspect.signature(self.model.fit)
        if "sample_weight" in sig.parameters and sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self._require("predict")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.isna().any().any():
            raise ValueError("NaN values present")
        if not all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
            raise TypeError("Non-numeric data present")
        return self.model.predict(df)

    def save(self, path: str | Path) -> str:
        joblib.dump(self.model, path)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> MLModel:
        return cls(joblib.load(path))

    @property
    def version(self) -> str:
        v = getattr(self.model, "version", None)
        return str(v) if v is not None else "unknown"

    # Backward-compatible alias used in some tests
    @property
    def pipeline(self):
        return self.model

    @pipeline.setter
    def pipeline(self, value):
        self.model = value


def save_model(model: Any, path: str | Path) -> None:
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)


def train_model(X: Any, y: Any, algorithm: str = "dummy") -> MLModel:
    if algorithm != "dummy":
        raise ValueError(f"unsupported algorithm: {algorithm}")
    if X is None or y is None:
        raise ValueError("invalid training data")
    return MLModel(_DummyPipe()).fit(X, y)


def predict_model(model: Any, X: Iterable | pd.DataFrame | None) -> list[float]:
    if model is None or X is None:
        raise ValueError("invalid input")
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    preds = model.predict(df)
    try:
        return list(preds)
    # noqa: BLE001 TODO: narrow exception
    except Exception:
        return []
