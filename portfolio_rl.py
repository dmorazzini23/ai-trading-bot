import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
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
