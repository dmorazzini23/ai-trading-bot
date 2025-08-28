"""Minimal torch stub used in tests when real PyTorch is unavailable."""
from __future__ import annotations

import sys
import types
from typing import Any, Iterable


class Optimizer:
    """No-op stand-in for :class:`torch.optim.Optimizer`."""

    def __init__(self, params: Iterable[Any] | None = None, lr: float = 0.001, **_: Any) -> None:
        self.params = list(params) if params is not None else []
        self.lr = lr

    def zero_grad(self) -> None:  # pragma: no cover - no-op
        return None

    def step(self) -> None:  # pragma: no cover - no-op
        return None


class _DummyOptimizer(Optimizer):
    """Base class for dummy optimizers that do nothing."""

    def __init__(self, params: Iterable[Any] | None = None, lr: float = 0.001, **kwargs: Any) -> None:
        super().__init__(params, lr, **kwargs)


class Adam(_DummyOptimizer):
    """Minimal stand-in for :class:`torch.optim.Adam`."""


class SGD(_DummyOptimizer):
    """Minimal stand-in for :class:`torch.optim.SGD`."""


# Expose a module-like "optim" namespace with the dummy classes
optim = types.ModuleType(__name__ + ".optim")
optim.Optimizer = Optimizer
optim.Adam = Adam
optim.SGD = SGD

# Register the submodule so ``import torch.optim`` works when this stub is
# inserted into ``sys.modules`` as ``torch``
sys.modules[__name__ + ".optim"] = optim
# Also map ``torch.optim`` for callers that alias this module to ``torch``
sys.modules.setdefault("torch.optim", optim)

__all__ = ["optim", "Optimizer", "Adam", "SGD"]
