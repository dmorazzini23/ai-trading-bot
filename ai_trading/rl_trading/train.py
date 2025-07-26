"""Train a PPO agent on historical data."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .env import TradingEnv

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - optional dependency
    PPO = None


def train(data: np.ndarray, model_path: str | Path, timesteps: int = 1000) -> str:
    if PPO is None:
        raise ImportError("stable-baselines3 required")
    env = TradingEnv(data)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(str(model_path))
    return str(model_path)
