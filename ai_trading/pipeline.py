# Core dependencies with graceful error handling
try:
    import numpy as np
except ImportError:
    print("WARNING: numpy not available in pipeline.py")
    np = None

import config

# ML dependencies with graceful error handling
try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    class BaseEstimator:
        pass
    
    class TransformerMixin:
        pass
    
    print("WARNING: sklearn not available, using basic object inheritance")

try:
    from sklearn.linear_model import SGDRegressor
except Exception:  # pragma: no cover - optional dependency
    class SGDRegressor:
        """Minimal stub when scikit-learn is unavailable."""

        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            if np is None:
                return [0] * len(X)
            return np.zeros(len(X))
try:
    from sklearn.pipeline import Pipeline
except Exception:  # pragma: no cover - optional dependency
    class Pipeline(list):
        """Simplistic pipeline fallback."""

        def __init__(self, steps):
            super().__init__(steps)

        def fit(self, X, y=None):
            for _, step in self:
                if hasattr(step, "fit"):
                    step.fit(X, y)
                if hasattr(step, "transform"):
                    X = step.transform(X)
            return self

        def predict(self, X):
            for name, step in self:
                if hasattr(step, "transform"):
                    X = step.transform(X)
            return self[-1][1].predict(X)
try:
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency
    class StandardScaler:
        def fit(self, X, y=None):
            if np is None:
                return self
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0) + 1e-9
            return self

        def transform(self, X):
            if np is None:
                return X
            return (X - self.mean_) / self.std_


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if hasattr(df, "columns"):
            close = df["close"].astype(float)
            # Calculate returns once and reuse
            returns = close.pct_change().fillna(0)

            features = {
                "returns": returns,
                "ma10": close.rolling(10).mean().bfill(),
                "ma30": close.rolling(30).mean().bfill(),
                "vol": returns.rolling(10).std().fillna(0),
                "sma_50": close.rolling(50).mean().bfill(),
                "sma_200": close.rolling(200).mean().bfill(),
                "price_change": (close.diff() > 0).astype(int),
                "rsi": self._calculate_rsi(close),  # Add RSI for better signals
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

    def _calculate_rsi(self, close, period=14):
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


model_pipeline = Pipeline(
    [
        ("features", FeatureBuilder()),
        ("scaler", StandardScaler()),
        ("regressor", SGDRegressor(warm_start=True, **config.SGD_PARAMS)),
    ]
)
