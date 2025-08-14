from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class _DummyPipe:
    """Pure-Python stand-in for an sklearn-like estimator."""
    fitted: bool = False
    version: str = "1"

    def fit(self, X: Iterable, y: Iterable) -> "_DummyPipe":
        self.fitted = True
        return self

    def predict(self, X: Iterable) -> np.ndarray:
        if not self.fitted:
            # mimic sklearn 'not fitted' shape of error
            raise AttributeError("not fitted")
        # return zeros with same length as X
        try:
            n = len(X)
        except Exception:
            n = 0
        return np.zeros(n)


class MLModel:
    """Lightweight wrapper with validation used in tests."""
    def __init__(self, pipe: Any | None = None):
        self.pipe = pipe or _DummyPipe()

    def fit(self, X: Sequence, y: Sequence) -> "MLModel":
        self.pipe.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # tests expect a ValueError when NaNs are present
        if isinstance(df, pd.DataFrame) and df.isna().any().any():
            raise ValueError("NaN values present")
        return self.pipe.predict(df)

    @property
    def version(self) -> str:
        v = getattr(self.pipe, "version", None)
        return str(v) if v is not None else "unknown"


# ----- simple persistence helpers (pickle) -----


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        # tests expect FileNotFoundError on missing file
        raise FileNotFoundError(str(path))
    with path.open("rb") as f:
        return pickle.load(f)


# ----- a trivial trainer for tests -----


def train_model(X: Any, y: Any, algorithm: str = "dummy") -> MLModel:
    # tests expect ValueError on bad algo or invalid data
    if algorithm not in {"dummy"}:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    if X is None or y is None:
        raise ValueError("invalid training data")
    model = MLModel(_DummyPipe())
    model.fit(X, y)
    return model
