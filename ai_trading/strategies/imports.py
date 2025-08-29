"""Lazy loaded imports for strategy-related dependencies."""

from __future__ import annotations

from ai_trading.logging import get_logger

logger = get_logger(__name__)

_NP = None
_PD = None
_METRICS = None
_RANDOM_FOREST_CLASSIFIER = None
_TRAIN_TEST_SPLIT = None
_TA = None

NUMPY_AVAILABLE = False
PANDAS_AVAILABLE = False
SKLEARN_AVAILABLE = False
TA_AVAILABLE = False


def get_np():
    """Return the :mod:`numpy` module on demand."""
    global _NP, NUMPY_AVAILABLE
    if _NP is None:
        import numpy as np  # pragma: no cover - import side effect

        _NP = np
        NUMPY_AVAILABLE = True
    return _NP


def get_pd():
    """Return the :mod:`pandas` module on demand."""
    global _PD, PANDAS_AVAILABLE
    if _PD is None:
        import pandas as pd  # pragma: no cover - import side effect

        _PD = pd
        PANDAS_AVAILABLE = True
    return _PD


def get_metrics():
    """Return :mod:`sklearn.metrics` lazily."""
    global _METRICS, SKLEARN_AVAILABLE
    if _METRICS is None:
        from sklearn import metrics  # pragma: no cover - import side effect

        _METRICS = metrics
        SKLEARN_AVAILABLE = True
    return _METRICS


def get_random_forest_classifier():
    """Return :class:`sklearn.ensemble.RandomForestClassifier` lazily."""
    global _RANDOM_FOREST_CLASSIFIER, SKLEARN_AVAILABLE
    if _RANDOM_FOREST_CLASSIFIER is None:
        from sklearn.ensemble import (  # pragma: no cover - import side effect
            RandomForestClassifier,
        )

        _RANDOM_FOREST_CLASSIFIER = RandomForestClassifier
        SKLEARN_AVAILABLE = True
    return _RANDOM_FOREST_CLASSIFIER


def get_train_test_split():
    """Return :func:`sklearn.model_selection.train_test_split` lazily."""
    global _TRAIN_TEST_SPLIT, SKLEARN_AVAILABLE
    if _TRAIN_TEST_SPLIT is None:
        from sklearn.model_selection import (  # pragma: no cover - import side effect
            train_test_split,
        )

        _TRAIN_TEST_SPLIT = train_test_split
        SKLEARN_AVAILABLE = True
    return _TRAIN_TEST_SPLIT


def get_ta():
    """Return the optional :mod:`ta` library when requested."""
    global _TA, TA_AVAILABLE
    if _TA is None:
        import ta  # pragma: no cover - import side effect

        logger.info(
            "TA library loaded successfully for enhanced technical analysis"
        )
        _TA = ta
        TA_AVAILABLE = True
    return _TA


__all__ = [
    "get_np",
    "get_pd",
    "get_metrics",
    "get_random_forest_classifier",
    "get_train_test_split",
    "get_ta",
    "NUMPY_AVAILABLE",
    "PANDAS_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "TA_AVAILABLE",
]

