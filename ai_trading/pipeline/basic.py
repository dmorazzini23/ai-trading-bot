"""Minimal pipeline utilities.

Provides a simple scikit-learn style pipeline with a basic
transformer that converts input data to a ``numpy`` array.
"""
from __future__ import annotations

import numpy as np

from ai_trading.logging import get_logger
from ai_trading.utils.lazy_imports import load_sklearn_pipeline

logger = get_logger(__name__)


class SimpleTransformer:
    """A no-op transformer used for tests and examples."""

    def fit(self, X, y=None):  # pragma: no cover - trivial
        self._fitted = True
        return self

    def transform(self, X):
        if not getattr(self, "_fitted", False):  # pragma: no cover - defensive
            raise RuntimeError("SimpleTransformer must be fitted before transform")
        return np.asarray(X, dtype=float)


def create_pipeline():
    """Create a basic pipeline with the :class:`SimpleTransformer`."""
    skl_pipe = load_sklearn_pipeline()
    if skl_pipe is None:  # pragma: no cover - runtime guard
        raise RuntimeError("sklearn.pipeline not available")
    Pipeline = skl_pipe.Pipeline
    return Pipeline([("simple", SimpleTransformer())])
