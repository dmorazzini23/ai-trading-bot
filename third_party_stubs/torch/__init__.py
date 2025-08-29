"""Minimal subset of the :mod:`torch` package used in tests.

The real PyTorch dependency is heavy and unnecessary for unit tests that only
verify integration with optional components like ``stable_baselines3``.  This
stub provides the handful of symbols imported by the project without offering
real tensor functionality.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from typing import Any

__version__ = "0.0.0"


class Tensor:
    """Placeholder tensor object."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        pass

    def to(self, *args: Any, **kwargs: Any) -> "Tensor":  # pragma: no cover - no-op
        return self

    def detach(self) -> "Tensor":  # pragma: no cover - no-op
        return self

    def cpu(self) -> "Tensor":  # pragma: no cover - no-op
        return self

    def numpy(self) -> Any:  # pragma: no cover - defensive
        raise ValueError("torch stub tensors have no backing array")


def as_tensor(data: Any, **kwargs: Any) -> Tensor:
    """Create a :class:`Tensor` from arbitrary data."""

    return Tensor()


def manual_seed(seed: int) -> None:  # pragma: no cover - deterministic
    """No-op seed function for API compatibility."""

    return None


class device:
    """Simplified device placeholder."""

    def __init__(self, type: str) -> None:  # pragma: no cover - trivial
        self.type = type


float32 = "float32"
float64 = "float64"


@contextmanager
def no_grad():  # pragma: no cover - context manager has no effect
    """Context manager that disables gradient tracking (noop)."""

    yield


class Module:
    """Base class for neural network modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


_nn = types.ModuleType("torch.nn")
_nn.Module = Module
sys.modules[__name__ + ".nn"] = _nn
nn = _nn

from . import optim  # noqa: F401  # imported for side effect to expose submodule

__all__ = [
    "Tensor",
    "as_tensor",
    "manual_seed",
    "device",
    "float32",
    "float64",
    "no_grad",
    "nn",
    "optim",
    "__version__",
]
