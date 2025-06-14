import os
import time
import logging
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

    def fit(self, X: pd.DataFrame, y) -> float:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        if len(X) == 0:
            self.logger.warning("TRAIN_SKIPPED_EMPTY_INPUT")
            return 0.0
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
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
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
            pipeline = joblib.load(path)
            logger.info("MODEL_LOADED", extra={"path": path})
        except Exception as exc:
            logger.exception(f"MODEL_LOAD_FAILED: {exc}")
            raise
        return cls(pipeline)
