"""Minimal sklearn stub for tests.

When the real :mod:`scikit-learn` package is installed this module will
defer to it so the application can use the full implementation.  The
lightweight stub defined here is only used when the dependency is missing
which is the case in the test environment.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType

_THIS_DIR = os.path.dirname(__file__)

def _load_real_sklearn() -> ModuleType | None:
    """Attempt to load the real scikit-learn package if installed."""
    for path in sys.path:
        if os.path.abspath(path) == os.path.abspath(_THIS_DIR):
            continue
        spec = importlib.machinery.PathFinder.find_spec(__name__, [path])
        if spec and spec.origin and os.path.abspath(os.path.dirname(spec.origin)) != os.path.abspath(_THIS_DIR):
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader
            sys.modules[__name__] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop(__name__, None)
                # If the real sklearn fails to import (e.g. incomplete install),
                # fall back to the lightweight stub rather than raising.
                return None
            return mod
    return None

_real = _load_real_sklearn()
if _real is not None:
    globals().update(_real.__dict__)
    sys.modules[__name__] = _real
    __all__ = getattr(_real, "__all__", [])
    base = sys.modules.get("sklearn.base", ModuleType("sklearn.base"))

    class BaseEstimator:
        """Minimal stand-in for :class:`sklearn.base.BaseEstimator`."""

        def __init__(self, *a, **k):
            pass

        def get_params(self, deep: bool = True):
            return {}

        def set_params(self, **params):
            return self

    class TransformerMixin:
        """Lightweight version of :class:`sklearn.base.TransformerMixin`."""

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    if not hasattr(base, "BaseEstimator"):
        base.BaseEstimator = BaseEstimator
    if not hasattr(base, "TransformerMixin"):
        base.TransformerMixin = TransformerMixin
    base.clone = getattr(base, "clone", lambda est: est)

    linear_model = sys.modules.get("sklearn.linear_model", ModuleType("sklearn.linear_model"))

    class Ridge:
        def __init__(self, *a, **k):
            pass

    class BayesianRidge:
        def __init__(self, *a, **k):
            pass

    if not hasattr(linear_model, "Ridge"):
        linear_model.Ridge = Ridge
    if not hasattr(linear_model, "BayesianRidge"):
        linear_model.BayesianRidge = BayesianRidge

    ensemble = sys.modules.get("sklearn.ensemble", ModuleType("sklearn.ensemble"))

    class RandomForestClassifier:
        def __init__(self, *a, **k):
            pass

    if not hasattr(ensemble, "RandomForestClassifier"):
        ensemble.RandomForestClassifier = RandomForestClassifier

    decomposition = sys.modules.get("sklearn.decomposition", ModuleType("sklearn.decomposition"))

    class PCA:
        def __init__(self, *a, **k):
            pass

    if not hasattr(decomposition, "PCA"):
        decomposition.PCA = PCA

    model_selection = sys.modules.get("sklearn.model_selection", ModuleType("sklearn.model_selection"))
    pipeline = sys.modules.get("sklearn.pipeline", ModuleType("sklearn.pipeline"))
    preprocessing = sys.modules.get("sklearn.preprocessing", ModuleType("sklearn.preprocessing"))

    for mod in [base, linear_model, ensemble, decomposition, model_selection, pipeline, preprocessing]:
        sys.modules.setdefault(mod.__name__, mod)

    __all__ = [
        "base",
        "linear_model",
        "ensemble",
        "decomposition",
        "model_selection",
        "pipeline",
        "preprocessing",
    ]
