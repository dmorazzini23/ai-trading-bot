"""Machine learning model training utilities."""

from __future__ import annotations

import json
import copy
from datetime import UTC, datetime
from typing import Any, TYPE_CHECKING
from pathlib import Path
from tempfile import gettempdir

import joblib
import numpy as np
from ai_trading.logging import logger
from ai_trading.models.artifacts import load_verified_joblib_artifact, write_artifact_manifest

# Optional dependencies
try:  # pragma: no cover - optional lightgbm dependency
    import lightgbm  # type: ignore  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover
    lightgbm = None  # type: ignore
    LIGHTGBM_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parents[2]
ALLOWED_DIRS = [BASE_DIR, Path(gettempdir()).resolve()]

try:  # pragma: no cover - optional dep
    import optuna

    optuna_available = True
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore
    optuna_available = False
from ..data.splits import PurgedGroupTimeSeriesSplit

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error


class MLTrainer:
    """
    Machine learning model trainer with purged cross-validation.

    Supports multiple model types with Optuna hyperparameter optimization
    and proper financial time series validation.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        cv_splits: int = 5,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.02,
        random_state: int = 42,
    ):
        """
        Initialize ML trainer.

        Args:
            model_type: Model type ('lightgbm', 'xgboost', 'ridge')
            cv_splits: Number of cross-validation splits
            embargo_pct: Embargo period percentage
            purge_pct: Purge period percentage
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.cv_splits = cv_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        self.random_state = random_state
        self.model: Any | None = None
        self.best_params: dict[str, Any] = {}
        self.cv_results: dict[str, Any] = {}
        self.feature_importance: dict[str, Any] = {}
        self.feature_pipeline: Any | None = None
        self._feature_count: int = 0
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Validate required dependencies are available."""
        if self.model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError(
                    "lightgbm is required for model_type 'lightgbm'. Install via `pip install lightgbm`.",
                )
        elif self.model_type == "xgboost":
            try:
                import xgboost  # noqa: F401
            except ImportError as exc:
                raise ImportError("XGBoost required for model_type 'xgboost'") from exc
        elif self.model_type == "ridge":
            try:
                import sklearn  # noqa: F401
            except ImportError as exc:
                raise ImportError("scikit-learn required for ridge model type") from exc
        elif self.model_type == "stacking":
            try:
                import sklearn  # noqa: F401
            except ImportError as exc:
                raise ImportError("scikit-learn required for stacking model type") from exc

    def train(
        self,
        X: "pd.DataFrame",
        y: "pd.Series",
        optimize_hyperparams: bool = True,
        optimization_trials: int = 100,
        feature_pipeline: Any | None = None,
        t1: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Train model with optional hyperparameter optimization.

        Args:
            X: Feature matrix
            y: Target series
            optimize_hyperparams: Whether to optimize hyperparameters
            optimization_trials: Number of Optuna trials
            feature_pipeline: Optional feature engineering pipeline
            t1: End times for purged CV (optional)

        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting {self.model_type} training with {len(X)} samples")
            cv_splitter = PurgedGroupTimeSeriesSplit(
                n_splits=self.cv_splits, embargo_pct=self.embargo_pct, purge_pct=self.purge_pct
            )
            if optimize_hyperparams and optuna_available:
                self.best_params = self._optimize_hyperparams(
                    X,
                    y,
                    cv_splitter,
                    optimization_trials,
                    t1,
                    feature_pipeline=feature_pipeline,
                )
            else:
                self.best_params = self._get_default_params()
            self.model = self._create_model(self.best_params)
            self.cv_results = self._evaluate_cv(
                X,
                y,
                cv_splitter,
                self.best_params,
                t1,
                feature_pipeline=feature_pipeline,
            )
            if (
                int(self.cv_results.get("n_splits", 0) or 0) <= 0
                or not np.isfinite(float(self.cv_results.get("mean_score", np.nan)))
            ):
                raise RuntimeError("CV evaluation produced no valid finite folds")
            self._fit_final_model(X, y, feature_pipeline=feature_pipeline)
            results = {
                "model_type": self.model_type,
                "best_params": self.best_params,
                "cv_metrics": self.cv_results,
                "feature_importance": self.feature_importance,
                "train_samples": len(X),
                "feature_count": self._feature_count or X.shape[1],
                "training_time": datetime.now(UTC).isoformat(),
            }
            logger.info(f"Training completed. CV score: {self.cv_results.get('mean_score', 'N/A')}")
            return results
        except (ValueError, TypeError) as e:
            logger.error(f"Error in model training: {e}")
            raise

    def _optimize_hyperparams(
        self,
        X: "pd.DataFrame",
        y: "pd.Series",
        cv_splitter: PurgedGroupTimeSeriesSplit,
        n_trials: int,
        t1: "pd.Series" | None = None,
        feature_pipeline: Any | None = None,
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        try:

            def objective(trial):
                params = self._suggest_params(trial)
                scores = []
                for train_idx, test_idx in cv_splitter.split(X, y, t1=t1):
                    X_train, X_test = (X.iloc[train_idx], X.iloc[test_idx])
                    y_train, y_test = (y.iloc[train_idx], y.iloc[test_idx])
                    X_train, X_test = self._fit_transform_fold(
                        X_train,
                        y_train,
                        X_test,
                        feature_pipeline,
                    )
                    model = self._create_model(params)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = self._calculate_score(y_test, predictions)
                    scores.append(score)
                return np.mean(scores)

            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state)
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            logger.info(f"Hyperparameter optimization completed. Best score: {study.best_value:.4f}")
            return dict(study.best_params)
        except (ValueError, TypeError) as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return self._get_default_params()

    def _suggest_params(self, trial) -> dict[str, Any]:
        """Suggest hyperparameters for different model types."""
        if self.model_type == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
        elif self.model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
        elif self.model_type == "ridge":
            return {
                "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"]),
            }
        else:
            return {}

    def _get_default_params(self) -> dict[str, Any]:
        """Get default parameters for different model types."""
        if self.model_type == "lightgbm":
            return {
                "n_estimators": 500,
                "learning_rate": 0.1,
                "max_depth": 6,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": self.random_state,
                "verbosity": -1,
            }
        elif self.model_type == "xgboost":
            return {
                "n_estimators": 500,
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "random_state": self.random_state,
            }
        elif self.model_type == "ridge":
            return {"alpha": 1.0, "fit_intercept": True, "solver": "auto", "random_state": self.random_state}
        elif self.model_type == "stacking":
            return {"meta_label_threshold": None}
        else:
            return {}

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create model instance with given parameters."""
        if self.model_type == "lightgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(**params)
        elif self.model_type == "xgboost":
            import xgboost as xgb

            return xgb.XGBRegressor(**params)
        elif self.model_type == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(**params)
        elif self.model_type == "stacking":
            from .stacking import StackingMetaModel

            return StackingMetaModel(
                cv_splits=self.cv_splits,
                embargo_pct=self.embargo_pct,
                purge_pct=self.purge_pct,
                meta_label_threshold=params.get("meta_label_threshold"),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _clone_feature_pipeline(self, feature_pipeline: Any) -> Any:
        """Clone a feature transformer for an isolated fold."""
        try:
            from sklearn.base import clone

            return clone(feature_pipeline)
        except (AttributeError, TypeError, ValueError):
            return copy.deepcopy(feature_pipeline)

    @staticmethod
    def _feature_names_from_transformer(transformer: Any, count: int) -> list[str]:
        get_names = getattr(transformer, "get_feature_names_out", None)
        if callable(get_names):
            try:
                return [str(name) for name in get_names()]
            except (TypeError, ValueError):
                pass
        return [f"feature_{i}" for i in range(count)]

    def _as_feature_frame(self, values: Any, *, index: Any, transformer: Any | None = None) -> Any:
        """Return transformed values as a DataFrame when possible."""
        import pandas as pd

        if isinstance(values, pd.DataFrame):
            return values
        if isinstance(values, pd.Series):
            return values.to_frame()
        columns = self._feature_names_from_transformer(transformer, values.shape[1])
        return pd.DataFrame(values, index=index, columns=columns)

    def _fit_transform_fold(
        self,
        X_train: "pd.DataFrame",
        y_train: "pd.Series",
        X_test: "pd.DataFrame",
        feature_pipeline: Any | None,
    ) -> tuple[Any, Any]:
        """Fit a feature transformer on one training fold and transform train/test."""
        if feature_pipeline is None:
            return X_train, X_test
        transformer = self._clone_feature_pipeline(feature_pipeline)
        if hasattr(transformer, "fit_transform"):
            X_train_processed = transformer.fit_transform(X_train, y_train)
        else:
            transformer.fit(X_train, y_train)
            X_train_processed = transformer.transform(X_train)
        transform = getattr(transformer, "transform", None)
        if not callable(transform):
            raise ValueError("feature_pipeline must implement transform for leakage-safe CV")
        X_test_processed = transform(X_test)
        return (
            self._as_feature_frame(X_train_processed, index=X_train.index, transformer=transformer),
            self._as_feature_frame(X_test_processed, index=X_test.index, transformer=transformer),
        )

    def _calculate_score(self, y_true: "pd.Series", y_pred: np.ndarray) -> float:
        """
        Calculate model score (Sharpe-like metric for trading).

        For trading models, we want to maximize risk-adjusted returns.
        """
        try:
            if np.all(np.abs(y_pred) < 10):
                pred_returns = y_pred
                true_returns = y_true.values
            else:
                pred_returns = y_pred / 100.0
                true_returns = y_true.values / 100.0
            directional_accuracy = np.mean(np.sign(pred_returns) == np.sign(true_returns))
            correlation = np.corrcoef(pred_returns, true_returns)[0, 1] if len(pred_returns) > 1 else 0.0
            if not np.isfinite(correlation):
                correlation = 0.0
            score = float(0.6 * directional_accuracy + 0.4 * correlation)
            return score
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating score: {e}")
            return 0.0

    def _evaluate_cv(
        self,
        X: "pd.DataFrame",
        y: "pd.Series",
        cv_splitter: PurgedGroupTimeSeriesSplit,
        params: dict[str, Any],
        t1: "pd.Series" | None = None,
        feature_pipeline: Any | None = None,
    ) -> dict[str, Any]:
        """Evaluate model using cross-validation."""
        try:
            from sklearn.metrics import mean_squared_error

            scores = []
            fold_results = []
            for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, t1=t1)):
                X_train, X_test = (X.iloc[train_idx], X.iloc[test_idx])
                y_train, y_test = (y.iloc[train_idx], y.iloc[test_idx])
                X_train, X_test = self._fit_transform_fold(
                    X_train,
                    y_train,
                    X_test,
                    feature_pipeline,
                )
                model = self._create_model(params)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                score = self._calculate_score(y_test, predictions)
                scores.append(score)
                mse = mean_squared_error(y_test, predictions)
                correlation = np.corrcoef(y_test, predictions)[0, 1] if len(predictions) > 1 else 0.0
                fold_results.append(
                    {
                        "fold": fold,
                        "score": score,
                        "mse": mse,
                        "correlation": correlation,
                        "train_samples": len(train_idx),
                        "test_samples": len(test_idx),
                    }
                )
            if not scores:
                return {
                    "mean_score": 0.0,
                    "std_score": 0.0,
                    "fold_scores": [],
                    "fold_details": fold_results,
                    "n_splits": 0,
                    "valid": False,
                    "error": "no_valid_cv_folds",
                }
            cv_results = {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "fold_scores": scores,
                "fold_details": fold_results,
                "n_splits": len(scores),
                "valid": True,
            }
            return cv_results
        except (ValueError, TypeError) as e:
            logger.error(f"Error in CV evaluation: {e}")
            return {"mean_score": 0.0, "std_score": 0.0, "fold_scores": []}

    def _fit_final_model(
        self,
        X: "pd.DataFrame",
        y: "pd.Series",
        feature_pipeline: Any | None = None,
    ) -> None:
        """Fit final model on full dataset."""
        try:
            if self.model is None:
                raise RuntimeError("Model must be initialized before fitting")
            feature_names = list(X.columns)
            fitted_model = self.model
            if feature_pipeline is not None:
                transformer = feature_pipeline
                self.feature_pipeline = transformer
                transform = getattr(transformer, "transform", None)
                if callable(transform):
                    from sklearn.pipeline import Pipeline

                    pipeline_model = self._create_model(self.best_params)
                    fitted_pipeline = Pipeline(
                        [
                            ("features", transformer),
                            ("model", pipeline_model),
                        ]
                    )
                    fitted_pipeline.fit(X, y)
                    self.model = fitted_pipeline
                    X_processed = self._as_feature_frame(
                        fitted_pipeline.named_steps["features"].transform(X),
                        index=X.index,
                        transformer=fitted_pipeline.named_steps["features"],
                    )
                    feature_names = list(X_processed.columns)
                    fitted_model = fitted_pipeline.named_steps["model"]
                else:
                    if hasattr(transformer, "fit_transform"):
                        X_processed = transformer.fit_transform(X, y)
                    else:
                        transformer.fit(X, y)
                        X_processed = transformer.transform(X)
                    X_processed = self._as_feature_frame(
                        X_processed,
                        index=X.index,
                        transformer=transformer,
                    )
                    feature_names = list(X_processed.columns)
                    self.model.fit(X_processed, y)
                    fitted_model = self.model
                self._feature_count = int(X_processed.shape[1])
            else:
                self.model.fit(X, y)
                self._feature_count = int(X.shape[1])
            if hasattr(fitted_model, "feature_importances_"):
                self.feature_importance = dict(zip(feature_names, fitted_model.feature_importances_, strict=False))
            elif hasattr(fitted_model, "coef_"):
                self.feature_importance = dict(zip(feature_names, np.abs(fitted_model.coef_), strict=False))
            logger.debug("Final model fitted successfully")
        except (ValueError, TypeError) as e:
            logger.error(f"Error fitting final model: {e}")
            raise

    def save_model(self, model_path: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Save trained model with metadata.

        Args:
            model_path: Path to save model (without extension)
            metadata: Additional metadata to save
        """
        try:
            model_file = Path(f"{model_path}.joblib").resolve()
            if not any(model_file.is_relative_to(d) for d in ALLOWED_DIRS):
                raise RuntimeError(f"Model path not allowed: {model_file}")
            joblib.dump(self.model, model_file)
            write_artifact_manifest(
                model_path=model_file,
                model_version="ml-trainer-v1",
                metadata={"model_type": self.model_type},
            )
            meta_data = {
                "model_type": self.model_type,
                "best_params": self.best_params,
                "cv_results": self.cv_results,
                "feature_importance": self.feature_importance,
                "training_timestamp": datetime.now(UTC).isoformat(),
                "random_state": self.random_state,
            }
            if metadata:
                meta_data.update(metadata)
            meta_file = model_file.with_name(model_file.stem + "_meta.json")
            with meta_file.open("w") as f:
                json.dump(meta_data, f, indent=2, default=str)
            logger.info(f"Model saved to {model_file}")
        except (OSError, ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, model_path: str) -> tuple[Any, dict[str, Any]]:
        """
        Load trained model with metadata.

        Args:
            model_path: Path to model file (without extension)

        Returns:
            Tuple of (model, metadata)
        """
        try:
            model_file = Path(f"{model_path}.joblib").resolve()
            if not any(model_file.is_relative_to(d) for d in ALLOWED_DIRS):
                raise RuntimeError(f"Model path not allowed: {model_file}")
            model = load_verified_joblib_artifact(model_file)
            meta_file = model_file.with_name(model_file.stem + "_meta.json")
            metadata = json.loads(meta_file.read_text())
            logger.info(f"Model loaded from {model_file}")
            return (model, metadata)
        except (OSError, ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Error loading model: {e}")
            raise


def train_model_cli(
    symbol_list: list[str], model_type: str = "lightgbm", dry_run: bool = False, wf_smoke: bool = False
) -> None:
    """
    CLI interface for model training.

    Args:
        symbol_list: List of symbols to train on
        model_type: Model type to train
        dry_run: Whether to run in dry-run mode
        wf_smoke: Whether to run walk-forward smoke test
    """
    try:
        normalized_symbols: list[str] = []
        seen_symbols: set[str] = set()
        for raw_symbol in symbol_list:
            symbol = str(raw_symbol or "").strip().upper()
            if not symbol or symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            normalized_symbols.append(symbol)
        if not normalized_symbols:
            raise ValueError("symbol_list must include at least one non-empty symbol")

        logger.info(f"CLI training started: symbols={normalized_symbols}, model={model_type}")
        if dry_run:
            logger.info("DRY RUN MODE - No actual training performed")
            return
        if wf_smoke:
            logger.info("Walk-forward smoke test - minimal training")
            trainer = MLTrainer(model_type=model_type, cv_splits=2)
            import pandas as pd

            np.random.seed(42)
            X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
            y = pd.Series(np.random.randn(100))
            results = trainer.train(X, y, optimize_hyperparams=False)
            logger.info(f"Smoke test completed: {results['cv_metrics']['mean_score']:.4f}")
        else:
            from ai_trading.model_loader import train_and_save_model
            from ai_trading.paths import MODELS_DIR

            trained_symbols: list[str] = []
            failed_symbols: dict[str, str] = {}
            logger.info(
                "Full training mode started",
                extra={"symbols": normalized_symbols, "models_dir": str(MODELS_DIR)},
            )
            for symbol in normalized_symbols:
                try:
                    train_and_save_model(symbol, MODELS_DIR)
                except (RuntimeError, ValueError, TypeError, ImportError) as exc:
                    failed_symbols[symbol] = str(exc)
                    logger.error(
                        "MODEL_TRAIN_FAILED",
                        extra={"symbol": symbol, "error": str(exc)},
                    )
                    continue
                trained_symbols.append(symbol)
                logger.info("MODEL_TRAINED", extra={"symbol": symbol})

            if not trained_symbols:
                raise RuntimeError(
                    f"Full ML training failed for all symbols: {normalized_symbols}"
                )
            logger.info(
                "Full training mode completed",
                extra={
                    "trained_symbols": trained_symbols,
                    "failed_symbols": failed_symbols,
                },
            )
    except (ValueError, TypeError) as e:
        logger.error(f"Error in CLI training: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Error in CLI training: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ML models for trading")
    parser.add_argument("--symbol-list", nargs="+", default=["AAPL", "MSFT"], help="List of symbols to train on")
    parser.add_argument(
        "--model-type", default="lightgbm", choices=["lightgbm", "xgboost", "ridge"], help="Model type to train"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--wf-smoke", action="store_true", help="Run walk-forward smoke test")
    args = parser.parse_args()
    train_model_cli(
        symbol_list=args.symbol_list, model_type=args.model_type, dry_run=args.dry_run, wf_smoke=args.wf_smoke
    )
