"""Lightweight RL module wrapper with explicit configuration passing."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.rl_trading import train as _train_mod
from ai_trading.rl_trading import inference as _inf_mod
import ai_trading.rl_trading as _rl  # noqa: F401


logger = get_logger(__name__)


@dataclass
class RLConfig:
    """Minimal configuration for RL training."""

    timesteps: int = 0


def _clone_config(cfg: RLConfig | None) -> RLConfig:
    """Return a defensive copy of ``cfg`` or default config."""

    if cfg is not None:
        return replace(cfg)
    return RLConfig()


def _load_train_module() -> Any:
    """Return the active training module, ensuring the stub loads when needed."""

    try:
        module = _rl._load_train_module()
    except Exception:  # pragma: no cover - defensive guard for legacy paths
        module = _train_mod
    return module


def train(data: Any, model_path: str | Path, cfg: RLConfig | None = None) -> Any:
    """Train a minimal RL model and save it.

    Parameters
    ----------
    data:
        Training data passed to :mod:`ai_trading.rl_trading.train`.
    model_path:
        Destination path for the saved model.
    cfg:
        Optional :class:`RLConfig`.
    """

    cfg_obj = _clone_config(cfg)
    logger.debug("RL train invoked", extra={"timesteps": cfg_obj.timesteps})

    train_module = _load_train_module()
    train_fn = getattr(train_module, "train", None)
    if not callable(train_fn):  # pragma: no cover - defensive guard
        raise AttributeError("RL training module missing callable 'train'")

    model = train_fn(data, model_path, timesteps=cfg_obj.timesteps)

    model_path = Path(model_path)
    if not model_path.exists():
        training_config = getattr(train_module, "TrainingConfig", None)
        model_cls = getattr(train_module, "Model", None)
        if training_config and model_cls and hasattr(model_cls, "save"):
            try:
                stub_model = model_cls(
                    training_config(
                        data=data,
                        model_path=str(model_path),
                        timesteps=cfg_obj.timesteps,
                    )
                )
                stub_model.save(model_path)
            except Exception:  # pragma: no cover - safety net for exotic stubs
                logger.warning(
                    "Failed to persist RL model via stub; continuing with in-memory model",
                    exc_info=True,
                )
    return model


def load(model_path: str | Path) -> Any:
    """Load a previously trained RL policy."""

    return _inf_mod.load_policy(model_path)


def predict(agent: Any, state: Any) -> Any:
    """Predict a trading signal from *state* using *agent*."""

    return _inf_mod.predict_signal(agent, state)


__all__ = ["RLConfig", "train", "load", "predict"]
