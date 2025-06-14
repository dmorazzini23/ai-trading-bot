import os
import time
import logging
from datetime import datetime
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error


class MLModel:
    """Wrapper around an sklearn Pipeline with extra safety checks."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.logger = logging.getLogger(__name__)

    def fit(self, X: pd.DataFrame, y) -> float:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        if len(X) == 0:
            self.logger.warning("TRAIN_SKIPPED_EMPTY_INPUT")
            return 0.0
        start = time.time()
        self.logger.info("MODEL_TRAIN_START", extra={"rows": len(X)})
        self.pipeline.fit(X, y)
        dur = time.time() - start
        preds = self.pipeline.predict(X)
        mse = float(mean_squared_error(y, preds))
        self.logger.info(
            "MODEL_TRAIN_END",
            extra={"duration": round(dur, 2), "mse": mse},
        )
        return mse

    def predict(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        preds = self.pipeline.predict(X)
        self.logger.info(
            "MODEL_PREDICT", extra={"rows": len(X)}
        )
        return preds

    def save(self, path: str | None = None) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path or os.path.join("models", f"model_{ts}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        self.logger.info("MODEL_SAVED", extra={"path": path})
        return path
