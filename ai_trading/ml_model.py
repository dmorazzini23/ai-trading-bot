"""Machine learning utilities for model training and inference.

Models are serialized with :mod:`joblib` and paths are validated to reside
within the local ``models`` directory.
"""

from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
import importlib.util
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from ai_trading import paths
from ai_trading.models.artifacts import (
    load_verified_joblib_artifact,
    write_artifact_manifest,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

try:  # pragma: no cover - optional heavy dependency
    import joblib  # type: ignore
    from joblib import parallel_backend

    with parallel_backend("loky", n_jobs=1):
        pass
except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover
    joblib = None  # type: ignore

    def parallel_backend(*args, **kwargs):  # type: ignore[override]
        class _Dummy:
            def __enter__(self) -> _Dummy:  # pragma: no cover - trivial
                return self

            def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
                return None

        return _Dummy()

logger = logging.getLogger(__name__)


def _require_pandas():
    try:
        return importlib.import_module("pandas")
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - optional dependency
        raise ImportError("pandas is required for ML model operations") from exc


def _require_joblib():
    if joblib is None:  # pragma: no cover - optional dependency
        raise ImportError("joblib is required for model persistence")
    return joblib


def _require_sklearn_linear_regression():
    try:
        from sklearn.linear_model import LinearRegression
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-learn is required for model training") from exc
    return LinearRegression


def train_model(
    X: Sequence[float] | Any,
    y: Sequence[float] | Any,
    algorithm: str = "linear",
) -> Any:
    """Train a simple linear model and return the estimator."""

    if X is None or y is None:
        raise ValueError("Invalid training data")
    if algorithm != "linear":
        raise ValueError("Unsupported algorithm")
    model = _require_sklearn_linear_regression()()
    model.fit([[v] for v in X], y)
    return model


def save_model(model: Any, path: str) -> None:
    """Persist ``model`` to ``path``."""

    model_dir = (Path(__file__).parent / "models").resolve()
    p = Path(path).resolve()
    if not p.is_relative_to(model_dir):
        raise RuntimeError(f"Model path outside allowed directory: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    _require_joblib().dump(model, p)
    write_artifact_manifest(
        model_path=p,
        model_version=str(getattr(model, "version", "ml-model-save-v1")),
        metadata={"source": "save_model"},
    )


def load_model(path: str) -> Any:
    """Load a model previously saved with ``save_model``."""

    model_dir = (Path(__file__).parent / "models").resolve()
    p = Path(path).resolve()
    if not p.is_relative_to(model_dir):
        raise RuntimeError(f"Model path outside allowed directory: {p}")
    if not p.exists():
        raise FileNotFoundError(p)
    return load_verified_joblib_artifact(p)


__all__ = [
    "train_model",
    "save_model",
    "load_model",
]
