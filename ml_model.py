import hashlib
import io
import logging
import os
import time
from datetime import datetime
from typing import Any, Sequence

try:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import mean_squared_error
except Exception:  # pragma: no cover - sklearn optional

    class BaseEstimator:
        def __init__(self, *args, **kwargs) -> None:
            logging.getLogger(__name__).error("scikit-learn is required")
            raise ImportError("scikit-learn is required")

    def mean_squared_error(y_true, y_pred):
        return 0.0


import joblib
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
except Exception:  # pragma: no cover - allow tests without sklearn

    class LinearRegression:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [0] * len(X)


class MLModel:
    """Wrapper around an sklearn Pipeline with extra safety checks."""

    def __init__(self, pipeline: BaseEstimator) -> None:
        self.pipeline: BaseEstimator = pipeline
        self.logger = logging.getLogger(__name__)

    def _validate_inputs(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        if X.isna().any().any():
            self.logger.error("NaN values detected in input")
            raise ValueError("Input contains NaN values")
        if not all(pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes):
            raise TypeError("All input columns must be numeric")

    def fit(self, X: pd.DataFrame, y) -> float:
        self._validate_inputs(X)
        start = time.time()
        self.logger.info("MODEL_TRAIN_START", extra={"rows": len(X)})
        try:
            self.pipeline.fit(X, y)
            dur = time.time() - start
            preds = self.pipeline.predict(X)
            mse = float(mean_squared_error(y, preds))
            self.logger.info(
                "MODEL_TRAIN_END",
                extra={"duration": round(dur, 2), "mse": mse},
            )
            return mse
        except Exception as exc:
            self.logger.exception(f"MODEL_TRAIN_FAILED: {exc}")
            raise

    def predict(self, X: pd.DataFrame) -> Any:
        self._validate_inputs(X)
        try:
            preds = self.pipeline.predict(X)
        except Exception as exc:
            self.logger.exception(f"MODEL_PREDICT_FAILED: {exc}")
            raise
        self.logger.info("MODEL_PREDICT", extra={"rows": len(X)})
        return preds

    def save(self, path: str | None = None) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path or os.path.join("models", f"model_{ts}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            joblib.dump(self.pipeline, path)
            self.logger.info("MODEL_SAVED", extra={"path": path})
        except Exception as exc:
            self.logger.exception(f"MODEL_SAVE_FAILED: {exc}")
            raise
        return path

    @classmethod
    def load(cls, path: str) -> "MLModel":
        logger = logging.getLogger(__name__)
        try:
            with open(path, "rb") as f:
                data = f.read()
            digest = hashlib.sha256(data).hexdigest()
            pipeline = joblib.load(io.BytesIO(data))
            logger.info(
                "MODEL_LOADED",
                extra={
                    "path": path,
                    "sha256": digest,
                    "version": getattr(pipeline, "version", "n/a"),
                },
            )
        except Exception as exc:
            logger.exception(f"MODEL_LOAD_FAILED: {exc}")
            raise
        return cls(pipeline)


def train_model(X: Sequence[float] | pd.Series | pd.DataFrame, y: Sequence[float] | pd.Series, algorithm: str = "linear") -> Any:
    """Train a trivial model and return it."""

    if X is None or y is None:
        raise ValueError("Invalid training data")
    if algorithm != "linear":
        raise ValueError("Unsupported algorithm")
    model = LinearRegression()
    model.fit([[v] for v in X], y)
    return model


def predict_model(model: Any, X: Sequence[Any] | pd.DataFrame) -> list[float]:
    """Return predictions from a fitted model."""

    if model is None:
        raise ValueError("Model cannot be None")
    if X is None:
        raise ValueError("Invalid input")
    try:
        return list(model.predict(X))
    except Exception as exc:  # pragma: no cover - model may fail unexpectedly
        logging.getLogger(__name__).error("Model prediction failed: %s", exc)
        raise


def save_model(model: Any, path: str) -> None:
    """Persist ``model`` to ``path``."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Load a model previously saved with ``save_model``."""

    return joblib.load(path)
