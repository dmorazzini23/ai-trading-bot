"""Reinforcement learning trading utilities."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict

from ai_trading.strategies.base import StrategySignal
TradeSignal = StrategySignal
logger = logging.getLogger(__name__)

_RL_STACK: Dict[str, Any] | None = None


def ensure_rl_stack() -> Dict[str, Any]:
    """Import and cache the reinforcement learning stack."""
    global _RL_STACK
    if _RL_STACK is not None:
        return _RL_STACK
    try:
        import gymnasium as gym
        import torch  # noqa: F401
        from stable_baselines3 import A2C, DQN, PPO
        from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # noqa: B904 - provide context
        raise ImportError(
            "stable-baselines3, gymnasium and torch are required for RL features"
        ) from exc
    _RL_STACK = {
        "gym": gym,
        "PPO": PPO,
        "A2C": A2C,
        "DQN": DQN,
        "BaseCallback": BaseCallback,
        "EvalCallback": EvalCallback,
        "make_vec_env": make_vec_env,
        "evaluate_policy": evaluate_policy,
        "DummyVecEnv": DummyVecEnv,
    }
    return _RL_STACK

class RLAgent:
    """Wrapper around a PPO policy for trading inference."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.model: Any | None = None

    def load(self) -> None:
        stack = ensure_rl_stack()
        if Path(self.model_path).exists():
            self.model = stack["PPO"].load(self.model_path)
        else:
            logger.error('RL model not found at %s', self.model_path)

    def predict(self, state, symbols: list[str] | None=None) -> TradeSignal | list[TradeSignal] | None:
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
            logger.error('RL model not loaded')
            return None
        try:
            if symbols is not None and hasattr(state, '__len__') and (len(state) == len(symbols)):
                actions, _ = self.model.predict(state, deterministic=True)
                signals: list[TradeSignal] = []
                for sym, act in zip(symbols, actions, strict=False):
                    side = {0: 'hold', 1: 'buy', 2: 'sell'}.get(int(act), 'hold')
                    signals.append(TradeSignal(symbol=sym, side=side, confidence=1.0, strategy='rl'))
                return signals
            action, _ = self.model.predict(state, deterministic=True)
            side = {0: 'hold', 1: 'buy', 2: 'sell'}.get(int(action), 'hold')
            return TradeSignal(symbol='RL', side=side, confidence=1.0, strategy='rl')
        except (KeyError, ValueError, TypeError) as exc:
            logger.error('RL prediction failed: %s', exc)
            return None

class RLTrader(RLAgent):
    """Backward-compatible alias used by bot_engine."""
    pass

try:
    __all__.extend(['RLTrader', 'ensure_rl_stack'])
except NameError:
    __all__ = ['RLAgent', 'RLTrader', 'ensure_rl_stack']
