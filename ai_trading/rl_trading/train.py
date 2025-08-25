"""Utilities for training reinforcement learning trading policies."""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ai_trading.exc import COMMON_EXC
from ai_trading.logging import logger

from . import ensure_rl_stack
from .env import TradingEnv


class RLTrainer:
    """Thin wrapper around Stable-Baselines3 algorithms."""

    def __init__(
        self,
        algorithm: str = "PPO",
        total_timesteps: int = 100_000,
        eval_freq: int = 10_000,
        seed: int = 42,
    ) -> None:
        stack = ensure_rl_stack()
        self.sb3: Dict[str, Any] = stack
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.seed = seed
        self.model: Any | None = None
        self.train_env: Any | None = None
        self.eval_env: Any | None = None

    # ------------------------------------------------------------------
    def train(
        self,
        data: np.ndarray,
        env_params: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train an RL policy on ``data``.

        Returns a dictionary with training metadata.
        """
        logger.info("Starting RL training using %s", self.algorithm)
        self._create_environments(data, env_params)
        self._create_model(model_params)
        start = datetime.now(UTC)
        callbacks = []  # Callbacks can be added here in the future.
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks, progress_bar=False)
        end = datetime.now(UTC)
        results = {
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "training_time_seconds": (end - start).total_seconds(),
            "seed": self.seed,
            "final_evaluation": self._final_evaluation(),
        }
        logger.info("RL training complete")
        return results

    # ------------------------------------------------------------------
    def _create_environments(self, data: np.ndarray, env_params: dict[str, Any] | None) -> None:
        env_params = env_params or {}
        split = int(len(data) * 0.8)
        train_data = data[:split]
        eval_data = data[split:]
        def make_train_env() -> TradingEnv:
            return TradingEnv(train_data, **env_params)
        stack = self.sb3
        self.train_env = stack["DummyVecEnv"]([make_train_env])
        self.eval_env = TradingEnv(eval_data, **env_params)

    # ------------------------------------------------------------------
    def _create_model(self, model_params: dict[str, Any] | None) -> None:
        params = {"seed": self.seed, "verbose": 0}
        if model_params:
            params.update(model_params)
        alg = self.algorithm.upper()
        if alg == "PPO":
            self.model = self.sb3["PPO"]("MlpPolicy", self.train_env, **params)
        elif alg == "A2C":
            self.model = self.sb3["A2C"]("MlpPolicy", self.train_env, **params)
        elif alg == "DQN":
            self.model = self.sb3["DQN"]("MlpPolicy", self.train_env, **params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    # ------------------------------------------------------------------
    def _final_evaluation(self) -> dict[str, float]:
        if self.model is None or self.eval_env is None:
            return {}
        try:
            mean, std = self.sb3["evaluate_policy"](
                self.model, self.eval_env, n_eval_episodes=5, deterministic=True
            )
            return {"mean_reward": float(mean), "std_reward": float(std)}
        except COMMON_EXC as exc:  # pragma: no cover - best effort
            logger.error("Evaluation failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model not trained")
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        self.model.save(str(path))


def train(
    data: np.ndarray,
    model_path: str | Path,
    *,
    timesteps: int = 100_000,
    algorithm: str = "PPO",
) -> RLTrainer:
    """Convenience wrapper that trains and saves a model."""
    trainer = RLTrainer(algorithm=algorithm, total_timesteps=timesteps)
    trainer.train(data)
    trainer.save(model_path)
    return trainer


def train_rl_model_cli() -> None:  # pragma: no cover - CLI helper
    """Run a demo training session using synthetic data."""
    try:
        stack = ensure_rl_stack()
        _ = stack  # noqa: F841 - import check only
    except ImportError as exc:
        logger.warning("RL stack unavailable: %s", exc)
        return
    np.random.seed(42)
    data = np.random.randn(1000, 4)
    trainer = RLTrainer(total_timesteps=10_000)
    results = trainer.train(data)
    logger.info("Training finished: %s", json.dumps(results))


if __name__ == "__main__":  # pragma: no cover
    train_rl_model_cli()

