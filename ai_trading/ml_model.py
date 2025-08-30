"""Machine learning utilities for model training and inference.

Models are serialized with :mod:`joblib` and paths are validated to reside
within the local ``models`` directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import logging
import pickle
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional heavy dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - fallback when pandas missing
    class _Series: ...  # noqa: D401 - simple placeholder

    class _DataFrame: ...

    class _PD:
        Series = _Series
        DataFrame = _DataFrame

    pd = _PD()  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    import joblib  # type: ignore
    from joblib import parallel_backend

    with parallel_backend("loky", n_jobs=1):
        pass
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

    def parallel_backend(*args, **kwargs):  # type: ignore[override]
        class _Dummy:
            def __enter__(self) -> _Dummy:  # pragma: no cover - trivial
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover
                return False

        return _Dummy()

logger = logging.getLogger(__name__)

if importlib.util.find_spec("sklearn") is not None:  # pragma: no cover - heavy
    from sklearn.base import BaseEstimator
    from sklearn.metrics import mean_squared_error
else:  # pragma: no cover

    class BaseEstimator:  # minimal fallback
        def __init__(self, *args, **kwargs) -> None:
            logger.error("scikit-learn is required")
            raise ImportError("scikit-learn is required")

    def mean_squared_error(y_true, y_pred):
        return 0.0

    logger.warning("scikit-learn not available; using fallback implementations")

if (
    importlib.util.find_spec("sklearn") is not None
    and importlib.util.find_spec("sklearn.linear_model") is not None
):  # pragma: no cover - heavy
    from sklearn.linear_model import LinearRegression
else:  # pragma: no cover

    class LinearRegression:  # minimal fallback
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [0] * len(X)

    logger.warning(
        "sklearn.linear_model not available; using fallback LinearRegression"
    )


class MLModel:
    """Wrapper around an sklearn Pipeline with extra safety checks."""

    def __init__(self, pipeline: BaseEstimator) -> None:
        self.pipeline: BaseEstimator = pipeline
        self.logger = logger

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_inputs(self, X: pd.DataFrame) -> None:
        import pandas as pd  # type: ignore

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        if X.isna().any().any():
            self.logger.error("NaN values detected in input")
            raise ValueError("Input contains NaN values")
        if not all((pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes)):
            raise TypeError("All input columns must be numeric")

    # ------------------------------------------------------------------
    # Model training and prediction
    # ------------------------------------------------------------------
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
                "MODEL_TRAIN_END", extra={"duration": round(dur, 2), "mse": mse}
            )
            return mse
        except (ValueError, RuntimeError) as exc:  # pragma: no cover - defensive
            self.logger.exception("MODEL_TRAIN_FAILED: %s", exc)
            raise

    def predict(self, X: pd.DataFrame) -> Any:
        self._validate_inputs(X)
        try:
            preds = self.pipeline.predict(X)
        except (ValueError, RuntimeError, AttributeError) as exc:  # pragma: no cover
            self.logger.exception("MODEL_PREDICT_FAILED: %s", exc)
            raise
        self.logger.info("MODEL_PREDICT", extra={"rows": len(X)})
        return preds

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str | None = None) -> str:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_dir = (Path(__file__).parent / "models").resolve()
        path = Path(path) if path else model_dir / f"model_{ts}.pkl"
        abs_path = path.resolve()
        if not abs_path.is_relative_to(model_dir):
            raise RuntimeError(f"Model path outside allowed directory: {abs_path}")
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if joblib is None:  # pragma: no cover - optional dep
            raise ImportError("joblib is required to save models")
        try:
            joblib.dump(self.pipeline, abs_path)
            self.logger.info("MODEL_SAVED", extra={"path": str(abs_path)})
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            self.logger.exception("MODEL_SAVE_FAILED: %s", exc)
            raise
        return str(abs_path)

    @classmethod
    def load(cls, path: str) -> MLModel:
        """Deserialize a saved model from ``path`` and return an ``MLModel``."""

        model_dir = (Path(__file__).parent / "models").resolve()
        abs_path = Path(path).resolve()
        if not abs_path.is_relative_to(model_dir):
            raise RuntimeError(f"Model path outside allowed directory: {abs_path}")
        if joblib is None:  # pragma: no cover - optional dep
            raise ImportError("joblib is required to load models")
        try:
            with abs_path.open("rb") as f:
                data = f.read()
            digest = hashlib.sha256(data).hexdigest()
            pipeline = joblib.load(io.BytesIO(data))
            logger.info(
                "MODEL_LOADED",
                extra={
                    "path": str(abs_path),
                    "sha256": digest,
                    "version": getattr(pipeline, "version", "n/a"),
                },
            )
        except (OSError, ValueError, pickle.UnpicklingError) as exc:  # pragma: no cover
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
    """Return predictions from a fitted model."""

    if model is None:
        raise ValueError("Model cannot be None")
    if X is None:
        raise ValueError("Invalid input")
    try:
        return list(model.predict(X))
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        logger.error("Model prediction failed: %s", exc)
        raise


def save_model(model: Any, path: str) -> None:
    """Persist ``model`` to ``path``."""

    model_dir = (Path(__file__).parent / "models").resolve()
    p = Path(path).resolve()
    if not p.is_relative_to(model_dir):
        raise RuntimeError(f"Model path outside allowed directory: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    if joblib is None:  # pragma: no cover - optional dep
        raise ImportError("joblib is required to save models")
    joblib.dump(model, p)


def load_model(path: str) -> Any:
    """Load a model previously saved with ``save_model``."""

    model_dir = (Path(__file__).parent / "models").resolve()
    p = Path(path).resolve()
    if not p.is_relative_to(model_dir):
        raise RuntimeError(f"Model path outside allowed directory: {p}")
    if joblib is None:  # pragma: no cover - optional dep
        raise ImportError("joblib is required to load models")
    return joblib.load(p)


def train_xgboost_with_optuna(
    X_train: Any, y_train: Any, X_val: Any, y_val: Any
) -> Any:  # pragma: no cover - heavy optional
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
        cv = xgb.cv(
            params, dtrain, num_boost_round=100, nfold=3, metrics="logloss"
        )
        return cv["test-logloss-mean"].iloc[-1]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model


__all__ = [
    "MLModel",
    "train_model",
    "predict_model",
    "save_model",
    "load_model",
    "train_xgboost_with_optuna",
]

