import numpy as np
import types

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    try:
        torch.SymInt
    except AttributeError:
        import logging
        logging.getLogger(__name__).warning("PyTorch version < 2.0 detected, using fallback implementation")
        torch = None
except Exception:  # pragma: no cover - optional dependency
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.tensor = lambda *a, **k: np.array([])
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    nn.Sequential = lambda *a, **k: None
    nn.Linear = lambda *a, **k: None
    nn.ReLU = lambda *a, **k: None
    nn.Softmax = lambda *a, **k: None
    optim = types.ModuleType("torch.optim")
    optim.Adam = lambda *a, **k: None


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        if torch is None:
            raise RuntimeError("PyTorch not available for reinforcement learning")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PortfolioReinforcementLearner:
    def __init__(self, state_dim: int = 10, action_dim: int = 5) -> None:
        if torch is None:
            raise RuntimeError("PyTorch not available for reinforcement learning")
        self.actor = Actor(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

    def rebalance_portfolio(self, state: np.ndarray) -> np.ndarray:
        if len(state) != 10:
            state = np.pad(state, (0, 10 - len(state)), "constant")
        state_tensor = torch.tensor(state, dtype=torch.float32)
        weights = self.actor(state_tensor).detach().numpy()
        total = weights.sum()
        if total == 0:
            total = 1.0
        return weights / total
