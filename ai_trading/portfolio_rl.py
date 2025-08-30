"""Simple reinforcement-learning portfolio manager.

This module defers importing :mod:`torch` until it's actually needed so that
environments without the optional dependency can still import the package. A
clear :class:`ImportError` is raised when functionality requires PyTorch.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache

from ai_trading.utils.device import TORCH_AVAILABLE


@lru_cache(maxsize=1)
def _lazy_import_torch():
    """Import :mod:`torch` and return its submodules."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for ai_trading.portfolio_rl")
    try:  # pragma: no cover - heavy optional dependency
        import torch as t
        from torch import nn as _nn, optim as _optim
    except (ImportError, OSError) as exc:  # pragma: no cover - import guard
        raise ImportError(
            "PyTorch is required for ai_trading.portfolio_rl",
        ) from exc
    return t, _nn, _optim


class Actor:
    """Wrapper around a small neural network policy."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        _, nn_mod, _ = _lazy_import_torch()
        self.net = nn_mod.Sequential(
            nn_mod.Linear(state_dim, 64),
            nn_mod.ReLU(),
            nn_mod.Linear(64, action_dim),
            nn_mod.Softmax(dim=-1),
        )

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()


class PortfolioReinforcementLearner:
    """Minimal RL-based portfolio balancer."""

    def __init__(self, state_dim: int = 10, action_dim: int = 5) -> None:
        _, _, optim_mod = _lazy_import_torch()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim)
        self.optimizer = optim_mod.Adam(self.actor.parameters(), lr=0.001)

    def rebalance_portfolio(self, state) -> np.ndarray:
        """Return normalized action weights for the given state."""

        torch_mod, _, _ = _lazy_import_torch()
        if hasattr(state, "tolist"):
            state = state.tolist()
        arr = np.asarray(state, dtype=np.float32)
        if arr.size < self.state_dim:
            arr = np.pad(arr, (0, self.state_dim - arr.size))
        elif arr.size > self.state_dim:
            arr = arr[: self.state_dim]
        state_tensor = torch_mod.tensor(arr, dtype=torch_mod.float32)
        with torch_mod.no_grad():
            weights = self.actor(state_tensor).numpy()
        total = float(weights.sum())
        return weights / total if total else weights


__all__ = ["PortfolioReinforcementLearner", "Actor"]

