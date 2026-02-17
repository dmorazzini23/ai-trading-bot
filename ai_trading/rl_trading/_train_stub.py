"""Fallback stub for :mod:`ai_trading.rl_trading.train`.

This lightweight module allows the RL training package to remain importable
when optional dependencies are unavailable.  The API mirrors the minimal
surface used in tests and production shims, ensuring callers can still obtain
deterministic placeholder behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger


logger = get_logger(__name__)


class _SB3Stub:
    """Minimal stand-in for Stable-Baselines3 policies."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def learn(self, *args: Any, **kwargs: Any) -> "_SB3Stub":
        return self

    def save(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    @classmethod
    def load(cls, *_args: Any, **_kwargs: Any) -> "_SB3Stub":
        return cls()


PPO = A2C = DQN = _SB3Stub


class BaseCallback:
    """Base callback placeholder used by the stub trainer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        pass


class EvalCallback(BaseCallback):
    """Evaluation callback placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        super().__init__(*args, **kwargs)


def make_vec_env(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - trivial
    return DummyVecEnv()


class DummyVecEnv(list):
    """Simple vectorised environment placeholder."""


def evaluate_policy(*_args: Any, **_kwargs: Any) -> tuple[float, float]:  # pragma: no cover - trivial
    return (0.0, 0.0)


@dataclass
class TrainingConfig:
    """Configuration for stub RL training."""

    data: Any | None = None
    model_path: str | os.PathLike[str] | None = None
    timesteps: int = 0


class Model:
    """Deterministic placeholder RL model."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def predict(self, _state: Any, deterministic: bool = True) -> tuple[int, None]:  # pragma: no cover - deterministic stub
        # Stub mode must remain non-trading by default.
        return (0, None)

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_bytes(b"0")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Model":
        return cls(TrainingConfig(model_path=str(path)))


def train(
    data: Any,
    model_path: str | os.PathLike[str],
    timesteps: int = 0,
) -> Model:
    """Train the stub RL model and persist a deterministic payload."""

    logger.warning(
        "Stable-Baselines3 unavailable; using RL training stub. "
        "Install the 'ai-trading-bot[rl]' extras for full functionality."
    )
    config = TrainingConfig(data=data, model_path=str(model_path), timesteps=timesteps)
    model = Model(config)
    Model.save(model, model_path)
    return model


__all__ = [
    "A2C",
    "BaseCallback",
    "DQN",
    "DummyVecEnv",
    "EvalCallback",
    "Model",
    "PPO",
    "TrainingConfig",
    "evaluate_policy",
    "make_vec_env",
    "train",
]
