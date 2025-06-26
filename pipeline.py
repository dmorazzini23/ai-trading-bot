import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


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
