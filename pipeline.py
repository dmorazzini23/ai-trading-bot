import numpy as np

import config
from sklearn.base import BaseEstimator, TransformerMixin
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
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0) + 1e-9
            return self

        def transform(self, X):
            return (X - self.mean_) / self.std_


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if hasattr(df, "columns"):
            close = df["close"].astype(float)
            features = {
                "returns": close.pct_change(fill_method=None).fillna(0),
                "ma10": close.rolling(10).mean().bfill(),
                "ma30": close.rolling(30).mean().bfill(),
                "vol": close.pct_change(fill_method=None).rolling(10).std().fillna(0),
                "sma_50": close.rolling(50).mean().bfill(),
                "sma_200": close.rolling(200).mean().bfill(),
                "price_change": (close.diff() > 0).astype(int),
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
                ]
            )
            return arr
        return np.asarray(df, dtype=float)


model_pipeline = Pipeline(
    [
        ("features", FeatureBuilder()),
        ("scaler", StandardScaler()),
        ("regressor", SGDRegressor(warm_start=True, **config.SGD_PARAMS)),
    ]
)
