"""Utilities providing a minimal ML model for tests.

This module exposes a small model class and a factory function used across
multiple tests. Keeping these objects in a dedicated module ensures they are
picklable by the standard library ``pickle`` module.
"""

from __future__ import annotations


class _DummyModel:
    """Trivial model with ``predict`` and ``predict_proba`` methods."""

    def predict(self, _x):  # pragma: no cover - trivial
        return [0]

    def predict_proba(self, _x):  # pragma: no cover - trivial
        return [[0.5, 0.5]]


def _get_model() -> _DummyModel:
    """Return an instance of the dummy model.

    Using a named function keeps the helper picklable with the standard
    library ``pickle`` module, whereas lambdas cannot be pickled.
    """

    return _DummyModel()


__all__ = ["_DummyModel", "_get_model"]
