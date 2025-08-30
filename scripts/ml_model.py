"""CLI wrapper for :mod:`ai_trading.ml_model`.

Legacy entrypoint that re-exports the package machine learning utilities.
"""

from ai_trading.ml_model import (  # noqa: F401
    MLModel,
    load_model,
    predict_model,
    save_model,
    train_model,
    train_xgboost_with_optuna,
)

__all__ = [
    "MLModel",
    "load_model",
    "predict_model",
    "save_model",
    "train_model",
    "train_xgboost_with_optuna",
]

if __name__ == "__main__":  # pragma: no cover - simple CLI hint
    print("Use ai_trading.ml_model instead")

