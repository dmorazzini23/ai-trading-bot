"""Minimal :mod:`torch.optim` stub for tests.

Only the pieces required by the codebase are implemented.  The classes do not
perform any optimisation but mirror the basic interface of their real
counterparts so that imports succeed.
"""
from __future__ import annotations

from typing import Any, Iterable, List


class Optimizer:
    """Very small subset of :class:`torch.optim.Optimizer`."""

    def __init__(self, params: Iterable[Any], lr: float | None = None) -> None:  # pragma: no cover - trivial
        self.param_groups: List[Any] = list(params)
        self.lr = lr

    def step(self) -> None:  # pragma: no cover - no-op
        return None

    def zero_grad(self) -> None:  # pragma: no cover - no-op
        return None


class Adam(Optimizer):
    """Placeholder for :class:`torch.optim.Adam`."""

    pass


__all__ = ["Optimizer", "Adam"]
