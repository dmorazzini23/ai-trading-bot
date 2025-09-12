"""Tiny dummy model used in tests."""
from __future__ import annotations


class _DummyModel:
    def predict(self, X):
        return [0] * len(X)


def _get_model() -> _DummyModel:
    return _DummyModel()


__all__ = ["_DummyModel", "_get_model"]

