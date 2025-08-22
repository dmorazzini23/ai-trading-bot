# Core dependencies
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ML dependencies
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


# SGD Parameters - reasonable defaults for financial time series
SGD_PARAMS = {
    'learning_rate': 'adaptive',
    'eta0': 0.01,
    'alpha': 0.0001,
    'max_iter': 1000,
    'tol': 1e-3,
    'random_state': 42
}

model_pipeline = Pipeline(
    [
        ("features", FeatureBuilder()),
        ("scaler", StandardScaler()),
        ("regressor", SGDRegressor(warm_start=True, **SGD_PARAMS)),
    ]
)
