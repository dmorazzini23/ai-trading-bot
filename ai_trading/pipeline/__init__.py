"""Machine learning model pipeline with lazy sklearn imports."""

from __future__ import annotations

import numpy as np

from ai_trading.logging import get_logger
from ai_trading.utils.lazy_imports import (
    load_sklearn_linear_model,
    load_sklearn_pipeline,
    load_sklearn_preprocessing,
)

logger = get_logger(__name__)


class FeatureBuilder:
    """Simple feature transformer for price-based data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if hasattr(df, "columns"):
            close = df["close"].astype(float)
            returns = close.pct_change().fillna(0)
            features = {
                "returns": returns,
                "ma10": close.rolling(10).mean().bfill(),
                "ma30": close.rolling(30).mean().bfill(),
                "vol": returns.rolling(10).std().fillna(0),
                "sma_50": close.rolling(50).mean().bfill(),
                "sma_200": close.rolling(200).mean().bfill(),
                "price_change": (close.diff() > 0).astype(int),
                "rsi": self._calculate_rsi(close),
            }
            arr = np.column_stack(
                [
                    features["returns"],
                    features["ma10"],
                    features["ma30"],
                    features["vol"],
                    features["sma_50"],
                    features["sma_200"],
                    features["price_change"],
                    features["rsi"],
                ]
            )
            return arr
        return np.asarray(df, dtype=float)

    def _calculate_rsi(self, close, period: int = 14):
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)


SGD_PARAMS = {
    "learning_rate": "adaptive",
    "eta0": 0.01,
    "alpha": 0.0001,
    "max_iter": 1000,
    "tol": 0.001,
    "random_state": 42,
}


class _LazyPipeline:
    _pipeline = None

    def _load(self):
        if self._pipeline is None:
            skl_pipeline = load_sklearn_pipeline()
            skl_pre = load_sklearn_preprocessing()
            skl_linear = load_sklearn_linear_model()
            if not all([skl_pipeline, skl_pre, skl_linear]):  # pragma: no cover - runtime guard
                raise RuntimeError("sklearn not available")
            Pipeline = skl_pipeline.Pipeline
            StandardScaler = skl_pre.StandardScaler
            SGDRegressor = skl_linear.SGDRegressor
            self._pipeline = Pipeline(
                [
                    ("features", FeatureBuilder()),
                    ("scaler", StandardScaler()),
                    ("regressor", SGDRegressor(warm_start=True, **SGD_PARAMS)),
                ]
            )
        return self._pipeline

    def __getattr__(self, item):  # pragma: no cover - passthrough
        return getattr(self._load(), item)


model_pipeline = _LazyPipeline()

__all__ = ["FeatureBuilder", "model_pipeline"]
