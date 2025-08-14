from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd

# AI-AGENT-REF: minimal ML model stub

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
    def __init__(self, pipe: Any | None = None):
        self.pipe = pipe or _DummyPipe()

    def fit(self, X: Sequence, y: Sequence) -> "MLModel":
        self.pipe.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if isinstance(df, pd.DataFrame) and df.isna().any().any():
            raise ValueError("NaN values present")
        return self.pipe.predict(df)

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.pipe, f)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "MLModel":
        with Path(path).open("rb") as f:
            pipe = pickle.load(f)
        return cls(pipe)

    @property
    def version(self) -> str:
        v = getattr(self.pipe, "version", None)
        return str(v) if v is not None else "unknown"


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("rb") as f:
        return pickle.load(f)


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
