import os
import time
import logging
import hashlib
import io
from datetime import datetime
from typing import Any

try:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import mean_squared_error
except Exception:  # pragma: no cover - sklearn optional
    class BaseEstimator:
        pass

    def mean_squared_error(y_true, y_pred):
        return 0.0

import joblib
import pandas as pd


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
                extra={"path": path, "sha256": digest, "version": getattr(pipeline, "version", "n/a")},
            )
        except Exception as exc:
            logger.exception(f"MODEL_LOAD_FAILED: {exc}")
            raise
        return cls(pipeline)
