"""Lightweight pipeline utilities for analysis tasks.

This module provides a minimal stand-in for :class:`sklearn.pipeline.Pipeline`
used in tests and simple analytics flows.  It supports sequential ``fit`` and
``transform`` operations without importing scikit-learn at module import time.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Tuple

from ai_trading.logging import get_logger

logger = get_logger(__name__)


class Pipeline:
    """Simplified pipeline for sequential transformations.

    Parameters
    ----------
    steps:
        Sequence of ``(name, transformer)`` pairs.  Each transformer may
        implement ``fit`` and ``transform`` methods.  The transformers are
        executed in the order provided.
    """

    def __init__(self, steps: Iterable[Tuple[str, Any]]):
        self.steps: List[Tuple[str, Any]] = list(steps)

    def fit(self, X, y: Any | None = None):
        """Fit each step sequentially and propagate transformations."""
        for _, step in self.steps[:-1]:
            if hasattr(step, "fit"):
                step.fit(X, y)
            if hasattr(step, "transform"):
                X = step.transform(X)
        last = self.steps[-1][1]
        if hasattr(last, "fit"):
            last.fit(X, y)
        return self

    def transform(self, X):
        """Return the transformed output of the final pipeline step.

        The data is passed through all intermediate ``transform`` steps before
        delegating to the final step's ``transform`` method.  This mirrors the
        behavior expected from ``sklearn.pipeline.Pipeline`` while keeping the
        implementation lightweight for testing.
        """
        for _, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                X = step.transform(X)
        last = self.steps[-1][1]
        if not hasattr(last, "transform"):
            raise AttributeError("Final pipeline step does not implement transform")
        return last.transform(X)


__all__ = ["Pipeline"]
