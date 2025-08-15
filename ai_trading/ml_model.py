from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class _DummyPipe:
    fitted: bool = False
    version: str = "1"

    def fit(self, X: Iterable, y: Iterable) -> "_DummyPipe":
        self.fitted = True
        return self

    def predict(self, X: Iterable) -> np.ndarray:
        if not self.fitted:
            raise AttributeError("not fitted")
        try:
            n = len(X)
        except Exception:
            n = 0
        return np.zeros(n)


class MLModel:
    def __init__(self, model: Any | None = None):
        self.pipeline = model or _DummyPipe()
        self._validate_model()

    def _validate_model(self) -> None:
        missing = [m for m in ("fit", "predict") if not hasattr(self.pipeline, m)]
        if missing:
            raise TypeError(
                f"Model missing required methods: {', '.join(missing)}"
            )

    def fit(self, X: Sequence, y: Sequence) -> "MLModel":
        return self.pipeline.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.isna().any().any():
            raise ValueError("NaN values present")
        if not all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes):
            raise TypeError("Non-numeric data present")
        return self.pipeline.predict(df)

    def save(self, path: str | Path) -> str:
        joblib.dump(self.pipeline, path)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "MLModel":
        pipeline = joblib.load(path)
        return cls(pipeline)

    @property
    def version(self) -> str:
        v = getattr(self.pipeline, "version", None)
        return str(v) if v is not None else "unknown"


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
    preds = model.predict(X)
    try:
        return list(preds)
    except Exception:
        return []
