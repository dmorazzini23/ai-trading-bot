"""Simple trading environment for RL agent."""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - optional dependency
    gym = None


class TradingEnv(gym.Env):  # type: ignore[misc]
    """Minimal gym environment for offline training."""

    def __init__(self, data: np.ndarray, window: int = 10):
        if gym is None:
            raise ImportError("gymnasium required")
        self.data = data.astype(np.float32)
        self.window = window
        self.current = window
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(window, data.shape[1]), dtype=np.float32
        )
        self.position = 0
        self.cash = 1.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current = self.window
        self.position = 0
        self.cash = 1.0
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        return self.data[self.current - self.window : self.current]

    def step(self, action: int):
        price = float(self.data[self.current, 0])
        reward = 0.0
        if action == 1:  # buy
            self.position += 1
            self.cash -= price
        elif action == 2 and self.position > 0:  # sell
            self.position -= 1
            self.cash += price
            reward = self.cash + self.position * price
        self.current += 1
        terminated = self.current >= len(self.data)
        return self._get_state(), reward, terminated, terminated, {}
