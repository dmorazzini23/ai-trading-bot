"""
Machine learning model training with purged cross-validation and Optuna optimization.

Provides training for LightGBM, XGBoost, and Ridge models with proper
financial time series validation and hyperparameter optimization.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# Use the centralized logger as per AGENTS.md
from ai_trading.logging import logger

# ML libraries are hard dependencies as per pyproject.toml
# AI-AGENT-REF: optional lightgbm shim for tests
try:
    import importlib
    lgb = importlib.import_module("lightgbm")
    lgb_available = True
except Exception:  # pragma: no cover - optional dependency
    from ai_trading.thirdparty import lightgbm_compat as lgb
    lgb_available = False

import xgboost as xgb

xgb_available = True

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
)

sklearn_available = True

import optuna

optuna_available = True

from ..data.splits import PurgedGroupTimeSeriesSplit


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

        # Model and results storage
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.feature_importance = None

        # Validation
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Validate required dependencies are available."""
        if self.model_type == "lightgbm" and not lgb_available:
            raise ImportError("LightGBM required for lightgbm model type")
        elif self.model_type == "xgboost" and not xgb_available:
            raise ImportError("XGBoost required for xgboost model type")
        elif self.model_type == "ridge" and not sklearn_available:
            raise ImportError("scikit-learn required for ridge model type")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
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

            # Apply feature pipeline if provided
            if feature_pipeline is not None:
                X_processed = feature_pipeline.fit_transform(X, y)
            else:
                X_processed = X.copy()

            # Create cross-validation splitter
            cv_splitter = PurgedGroupTimeSeriesSplit(
                n_splits=self.cv_splits,
                embargo_pct=self.embargo_pct,
                purge_pct=self.purge_pct,
            )

            # Hyperparameter optimization
            if optimize_hyperparams and optuna_available:
                self.best_params = self._optimize_hyperparams(
                    X_processed, y, cv_splitter, optimization_trials, t1
                )
            else:
                self.best_params = self._get_default_params()

            # Train final model with best parameters
            self.model = self._create_model(self.best_params)

            # Cross-validation evaluation
            self.cv_results = self._evaluate_cv(
                X_processed, y, cv_splitter, self.best_params, t1
            )

            # Train on full dataset
            self._fit_final_model(X_processed, y)

            # Generate results
            results = {
                "model_type": self.model_type,
                "best_params": self.best_params,
                "cv_metrics": self.cv_results,
                "feature_importance": self.feature_importance,
                "train_samples": len(X),
                "feature_count": X_processed.shape[1],
                "training_time": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                f"Training completed. CV score: {self.cv_results.get('mean_score', 'N/A')}"
            )
            return results

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def _optimize_hyperparams(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splitter: PurgedGroupTimeSeriesSplit,
        n_trials: int,
        t1: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        try:

            def objective(trial):
                params = self._suggest_params(trial)
                scores = []

                for train_idx, test_idx in cv_splitter.split(X, y, t1=t1):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    model = self._create_model(params)
                    model.fit(X_train, y_train)

                    predictions = model.predict(X_test)
                    score = self._calculate_score(y_test, predictions)
                    scores.append(score)

                return np.mean(scores)

            # Create study
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )

            # Optimize
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            logger.info(
                f"Hyperparameter optimization completed. Best score: {study.best_value:.4f}"
            )
            return study.best_params

        except Exception as e:
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
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
                "solver": trial.suggest_categorical(
                    "solver", ["auto", "svd", "cholesky", "lsqr"]
                ),
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
            return {
                "alpha": 1.0,
                "fit_intercept": True,
                "solver": "auto",
                "random_state": self.random_state,
            }
        else:
            return {}

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create model instance with given parameters."""
        if self.model_type == "lightgbm":
            return lgb.LGBMRegressor(**params)
        elif self.model_type == "xgboost":
            return xgb.XGBRegressor(**params)
        elif self.model_type == "ridge":
            return Ridge(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Calculate model score (Sharpe-like metric for trading).

        For trading models, we want to maximize risk-adjusted returns.
        """
        try:
            # Convert predictions to returns if needed
            if np.all(np.abs(y_pred) < 10):  # Assume these are returns
                pred_returns = y_pred
                true_returns = y_true.values
            else:
                # Convert to returns
                pred_returns = y_pred / 100.0
                true_returns = y_true.values / 100.0

            # Calculate directional accuracy
            directional_accuracy = np.mean(
                np.sign(pred_returns) == np.sign(true_returns)
            )

            # Calculate correlation
            correlation = np.corrcoef(pred_returns, true_returns)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            # Combine metrics (emphasize both accuracy and correlation)
            score = 0.6 * directional_accuracy + 0.4 * abs(correlation)

            return score

        except Exception as e:
            logger.error(f"Error calculating score: {e}")
            return 0.0

    def _evaluate_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splitter: PurgedGroupTimeSeriesSplit,
        params: dict[str, Any],
        t1: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Evaluate model using cross-validation."""
        try:
            scores = []
            fold_results = []

            for fold, (train_idx, test_idx) in enumerate(
                cv_splitter.split(X, y, t1=t1)
            ):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
                model = self._create_model(params)
                model.fit(X_train, y_train)

                # Evaluate
                predictions = model.predict(X_test)
                score = self._calculate_score(y_test, predictions)
                scores.append(score)

                # Calculate additional metrics
                mse = mean_squared_error(y_test, predictions)
                correlation = (
                    np.corrcoef(y_test, predictions)[0, 1]
                    if len(predictions) > 1
                    else 0.0
                )

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

            # Aggregate results
            cv_results = {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "fold_scores": scores,
                "fold_details": fold_results,
                "n_splits": len(scores),
            }

            return cv_results

        except Exception as e:
            logger.error(f"Error in CV evaluation: {e}")
            return {"mean_score": 0.0, "std_score": 0.0, "fold_scores": []}

    def _fit_final_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit final model on full dataset."""
        try:
            self.model.fit(X, y)

            # Extract feature importance if available
            if hasattr(self.model, "feature_importances_"):
                self.feature_importance = dict(
                    zip(X.columns, self.model.feature_importances_, strict=False)
                )
            elif hasattr(self.model, "coef_"):
                self.feature_importance = dict(
                    zip(X.columns, np.abs(self.model.coef_), strict=False)
                )

            logger.debug("Final model fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting final model: {e}")
            raise

    def save_model(
        self, model_path: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Save trained model with metadata.

        Args:
            model_path: Path to save model (without extension)
            metadata: Additional metadata to save
        """
        try:
            # Save model
            model_file = f"{model_path}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(self.model, f)

            # Create metadata
            meta_data = {
                "model_type": self.model_type,
                "best_params": self.best_params,
                "cv_results": self.cv_results,
                "feature_importance": self.feature_importance,
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                "random_state": self.random_state,
            }

            if metadata:
                meta_data.update(metadata)

            # Save metadata
            meta_file = f"{model_path}_meta.json"
            with open(meta_file, "w") as f:
                json.dump(meta_data, f, indent=2, default=str)

            logger.info(f"Model saved to {model_file}")

        except Exception as e:
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
            # Load model
            model_file = f"{model_path}.pkl"
            with open(model_file, "rb") as f:
                model = pickle.load(f)

            # Load metadata
            meta_file = f"{model_path}_meta.json"
            with open(meta_file) as f:
                metadata = json.load(f)

            logger.info(f"Model loaded from {model_file}")
            return model, metadata

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def train_model_cli(
    symbol_list: list[str],
    model_type: str = "lightgbm",
    dry_run: bool = False,
    wf_smoke: bool = False,
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
        logger.info(f"CLI training started: symbols={symbol_list}, model={model_type}")

        if dry_run:
            logger.info("DRY RUN MODE - No actual training performed")
            return

        if wf_smoke:
            logger.info("Walk-forward smoke test - minimal training")
            # Simplified training for smoke test
            trainer = MLTrainer(model_type=model_type, cv_splits=2)

            # Create dummy data for smoke test
            np.random.seed(42)
            X = pd.DataFrame(
                np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)]
            )
            y = pd.Series(np.random.randn(100))

            results = trainer.train(X, y, optimize_hyperparams=False)
            logger.info(
                f"Smoke test completed: {results['cv_metrics']['mean_score']:.4f}"
            )
        else:
            logger.info("Full training mode would require actual market data")
            logger.info("Placeholder for full training implementation")

    except Exception as e:
        logger.error(f"Error in CLI training: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ML models for trading")
    parser.add_argument(
        "--symbol-list",
        nargs="+",
        default=["AAPL", "MSFT"],
        help="List of symbols to train on",
    )
    parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=["lightgbm", "xgboost", "ridge"],
        help="Model type to train",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument(
        "--wf-smoke", action="store_true", help="Run walk-forward smoke test"
    )

    args = parser.parse_args()

    train_model_cli(
        symbol_list=args.symbol_list,
        model_type=args.model_type,
        dry_run=args.dry_run,
        wf_smoke=args.wf_smoke,
    )
