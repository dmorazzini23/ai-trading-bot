from __future__ import annotations
import numpy as np
# AI-AGENT-REF: lightgbm shim for tests


class _StubBooster:
    def predict(self, X):
        try:
            n = len(X)
        except Exception:
            n = 0
        return np.zeros(n)


class LGBMClassifier:  # minimal surface for tests
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):  # pragma: no cover - trivial stub
        return self

    def predict(self, X):  # pragma: no cover - trivial stub
        return _StubBooster().predict(X)
