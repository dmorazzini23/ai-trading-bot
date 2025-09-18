"""Lightweight heuristic ML model with scikit-like API.

This provides a safe default when no serialized model artifact is available.
It consumes the indicator features already computed by the bot and emits a
binary prediction with a rough probability.

API:
- predict(X) -> ndarray[int]
- predict_proba(X) -> ndarray[ [p0, p1] ]

Feature contract (order matters):
    ["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"]
"""

from __future__ import annotations

from typing import Any
import numpy as np


class HeuristicModel:
    """Lightweight heuristic classifier acting as a placeholder model."""

    is_placeholder_model = True
    classes_ = np.array([0, 1], dtype=int)
    feature_names_in_ = np.array(["rsi", "macd", "atr", "vwap", "sma_50", "sma_200"])  # type: ignore[attr-defined]

    def _score_row(self, x: np.ndarray) -> float:
        # x order: rsi, macd, atr, vwap, sma_50, sma_200
        rsi, macd, atr, vwap, sma50, sma200 = x.astype(float)
        score = 0.0
        # RSI above 55 is moderately bullish; below 45 is bearish
        score += (rsi - 50.0) / 20.0  # ~[-2.5, +2.5]
        # Positive MACD is bullish; scale down to reasonable range
        score += np.tanh(macd)  # [-1,1]
        # Price above moving averages is bullish
        score += 0.5 if vwap > sma50 else -0.5
        score += 0.3 if vwap > sma200 else -0.3
        # Higher ATR -> lower confidence; dampen score slightly
        score -= 0.1 * np.tanh(atr)
        return score

    def predict(self, X: Any) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        scores = np.apply_along_axis(self._score_row, 1, X)
        return (scores > 0.0).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        scores = np.apply_along_axis(self._score_row, 1, X)
        # Logistic squashing to [0,1]
        p1 = 1.0 / (1.0 + np.exp(-scores))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


def get_model() -> HeuristicModel:
    return HeuristicModel()


__all__ = ["HeuristicModel", "get_model"]

