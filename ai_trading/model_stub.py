from __future__ import annotations

"""Example model module providing a lightweight linear model."""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Model:
    """Simple linear model using predefined coefficients."""

    coefficients: list[float]

    def predict(self, features: Iterable[float]) -> float:
        """Return a single prediction for ``features``."""
        return float(sum(c * f for c, f in zip(self.coefficients, features)))


def get_model() -> Model:
    """Return a :class:`Model` instance with example coefficients."""
    return Model(coefficients=[0.1, 0.2, 0.3])
