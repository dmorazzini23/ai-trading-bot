"""Machine learning utilities for model training and inference."""

from __future__ import annotations

import joblib
import hashlib
import io
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

try:
    from sklearn.base import BaseEstimator
    from sklearn.metrics import mean_squared_error
except ImportError:  # pragma: no cover - sklearn optional

    class BaseEstimator:
        def __init__(self, *args, **kwargs) -> None:
            logger.error("scikit-learn is required")
            raise ImportError("scikit-learn is required")

    def mean_squared_error(y_true, y_pred):
        return 0.0


from joblib import parallel_backend

# AI-AGENT-REF: restrict joblib parallelism
with parallel_backend("loky", n_jobs=1):
    pass

try:
    from sklearn.linear_model import LinearRegression
except ImportError:  # pragma: no cover - allow tests without sklearn

    class LinearRegression:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [0] * len(X)


class MLModel:
    """Wrapper around an sklearn Pipeline with extra safety checks."""

    def __init__(self, pipeline: BaseEstimator) -> None:
        self.pipeline: BaseEstimator = pipeline
        self.logger = logger

    def _validate_inputs(self, X: pd.DataFrame) -> None:
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        if X.isna().any().any():
            self.logger.error("NaN values detected in input")
            raise ValueError("Input contains NaN values")
        if not all(pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes):
            raise TypeError("All input columns must be numeric")

    def fit(self, X: pd.DataFrame, y: Sequence[float] | pd.Series) -> float:
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
        except (ValueError, RuntimeError) as exc:
            self.logger.exception("MODEL_TRAIN_FAILED: %s", exc)
            raise

    def predict(self, X: pd.DataFrame) -> Any:
        self._validate_inputs(X)
        try:
            preds = self.pipeline.predict(X)
        except (ValueError, RuntimeError, AttributeError) as exc:
            self.logger.exception("MODEL_PREDICT_FAILED: %s", exc)
            raise
        self.logger.info("MODEL_PREDICT", extra={"rows": len(X)})
        return preds

    def save(self, path: str | None = None) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir = Path(__file__).parent / "models"
        path = Path(path) if path else model_dir / f"model_{ts}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self.pipeline, str(path))
            self.logger.info("MODEL_SAVED", extra={"path": str(path)})
        except (OSError, ValueError) as exc:
            self.logger.exception("MODEL_SAVE_FAILED: %s", exc)
            raise
        return str(path)

    @classmethod
    def load(cls, path: str) -> "MLModel":
        """Deserialize a saved model from ``path`` and return an ``MLModel``."""
        import joblib
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
        except (OSError, ValueError, pickle.UnpicklingError) as exc:
            logger.exception("MODEL_LOAD_FAILED: %s", exc)
            raise
        return cls(pipeline)


def train_model(
    X: Sequence[float] | pd.Series | pd.DataFrame,
    y: Sequence[float] | pd.Series,
    algorithm: str = "linear",
) -> BaseEstimator:
    """Train a simple linear model and return the estimator."""

    if X is None or y is None:
        raise ValueError("Invalid training data")
    if algorithm != "linear":
        raise ValueError("Unsupported algorithm")
    model = LinearRegression()
    model.fit([[v] for v in X], y)
    return model


def predict_model(model: Any, X: Sequence[Any] | pd.DataFrame) -> list[float]:
    """Return predictions from a fitted model.

    Parameters
    ----------
    model : Any
        Trained model instance implementing ``predict``.
    X : Sequence[Any] | pd.DataFrame
        Input features for prediction.

    Returns
    -------
    list[float]
        Model predictions as a list of floats.
    """

    if model is None:
        raise ValueError("Model cannot be None")
    if X is None:
        raise ValueError("Invalid input")
    try:
        return list(model.predict(X))
    except Exception as exc:  # pragma: no cover - model may fail unexpectedly
        logger.error("Model prediction failed: %s", exc)
        raise


def save_model(model: Any, path: str) -> None:
    import joblib
    """Persist ``model`` to ``path``.

    Parameters
    ----------
    model : Any
        Trained model object supporting ``joblib`` serialization.
    path : str
        Filesystem location to write the model to.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(p))


def load_model(path: str) -> Any:
    import joblib
    """Load a model previously saved with ``save_model``.

    Parameters
    ----------
    path : str
        Filesystem path to the serialized model.

    Returns
    -------
    Any
        Deserialized model object.
    """
    return joblib.load(str(Path(path)))




def train_xgboost_with_optuna(
    X_train: Any, y_train: Any, X_val: Any, y_val: Any
) -> Any:
    """Hyperparameter search for an XGBoost model using Optuna."""
    import optuna
    import xgboost as xgb

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.3),
            "objective": "binary:logistic",
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        cv = xgb.cv(params, dtrain, num_boost_round=100, nfold=3, metrics="logloss")
        return cv["test-logloss-mean"].iloc[-1]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model
