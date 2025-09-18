"""Lightweight RL module wrapper with optional config alias ``_C``.

This shim exposes minimal training and inference helpers while avoiding a
hard dependency on the full RL stack.  Downstream code may import
``ai_trading.rl.module`` and optionally access the ``_C`` configuration
object, but the functions will operate even if ``_C`` is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
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


# Backwards-compatibility alias.  Older code expected a module level ``_C``
# configuration similar to other subsystems.  New code should pass
# ``RLConfig`` explicitly, but ``_C`` is kept for legacy access.
_C = RLConfig()


def train(data: Any, model_path: str | Path, cfg: RLConfig | None = None):
    """Train a minimal RL model and save it.

    Parameters
    ----------
    data:
        Training data passed to :mod:`ai_trading.rl_trading.train`.
    model_path:
        Destination path for the saved model.
    cfg:
        Optional :class:`RLConfig` overriding the legacy ``_C`` defaults.
    """

    # ``_C`` may be removed by callers to avoid the legacy alias.  Fall back to
    # a fresh configuration if the global alias is absent.
    cfg = cfg or globals().get("_C") or RLConfig()
    logger.debug("RL train invoked", extra={"timesteps": cfg.timesteps})
    return _train_mod.train(data, model_path, timesteps=cfg.timesteps)


def load(model_path: str | Path):
    """Load a previously trained RL policy."""

    return _inf_mod.load_policy(model_path)


def predict(agent: Any, state: Any):
    """Predict a trading signal from *state* using *agent*."""

    return _inf_mod.predict_signal(agent, state)


__all__ = ["RLConfig", "train", "load", "predict", "_C"]
