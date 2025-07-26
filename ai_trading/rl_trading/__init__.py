"""Reinforcement learning trading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore

from strategies.base import TradeSignal

logger = logging.getLogger(__name__)


class RLAgent:
    """Wrapper around a PPO policy for trading inference."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.model: Any | None = None

    def load(self) -> None:
        if PPO is None:
            raise ImportError("stable-baselines3 required")
        if Path(self.model_path).exists():
            self.model = PPO.load(self.model_path)
        else:  # pragma: no cover - model optional in tests
            logger.error("RL model not found at %s", self.model_path)

    def predict(self, state, symbols: list[str] | None = None) -> TradeSignal | list[TradeSignal] | None:
        """
        Predict one or more trade signals from the current model.

        Parameters
        ----------
        state : Any
            Observation or batch of observations passed to the underlying policy.
        symbols : list[str] | None, optional
            When provided, a list of symbols corresponding to each state row.
            If omitted, a single generic signal is returned.

        Returns
        -------
        TradeSignal | list[TradeSignal] | None
            A single trade signal or a list of signals (one per symbol).
        """
        if self.model is None:
            logger.error("RL model not loaded")
            return None
        try:
            # batch prediction when multiple symbols are provided
            if symbols is not None and hasattr(state, "__len__") and len(state) == len(symbols):
                actions, _ = self.model.predict(state, deterministic=True)
                signals: list[TradeSignal] = []
                for sym, act in zip(symbols, actions):
                    side = {0: "hold", 1: "buy", 2: "sell"}.get(int(act), "hold")
                    signals.append(TradeSignal(symbol=sym, side=side, confidence=1.0, strategy="rl"))
                return signals
            # single prediction fallback
            action, _ = self.model.predict(state, deterministic=True)
            side = {0: "hold", 1: "buy", 2: "sell"}.get(int(action), "hold")
            return TradeSignal(symbol="RL", side=side, confidence=1.0, strategy="rl")
        except Exception as exc:
            logger.error("RL prediction failed: %s", exc)
            return None
