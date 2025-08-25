"""Reinforcement learning trading utilities with optional dependencies."""
from __future__ import annotations

from functools import lru_cache
import importlib
import logging
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Exposed for tests to monkeypatch
PPO: Any | None = None
DummyVecEnv: Any | None = None

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from ai_trading.strategies.base import StrategySignal  # noqa: F401


@lru_cache(maxsize=1)
def _load_rl_stack() -> dict[str, Any] | None:
    """Attempt to import the optional RL stack and cache the result."""
    try:
        sb3 = importlib.import_module("stable_baselines3")
        gym = importlib.import_module("gymnasium")
        importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001 - best-effort import
        logger.debug("RL stack unavailable: %s", exc)
        return None
    global PPO, DummyVecEnv
    PPO = sb3.PPO
    DummyVecEnv = sb3.common.vec_env.DummyVecEnv
    return {"sb3": sb3, "gym": gym}


def is_rl_available() -> bool:
    """Return True if the optional RL dependencies can be imported."""
    return _load_rl_stack() is not None


class RLAgent:
    """Wrapper around a PPO policy for trading inference."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.model: Any | None = None

    def load(self) -> None:
        if not is_rl_available() or PPO is None:
            raise ImportError("stable-baselines3 required")
        if Path(self.model_path).exists():
            self.model = PPO.load(self.model_path)
        else:
            logger.error("RL model not found at %s", self.model_path)

    def predict(
        self, state, symbols: list[str] | None = None
    ) -> "StrategySignal" | list["StrategySignal"] | None:
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
        from ai_trading.strategies.base import StrategySignal  # noqa: E402

        try:
            if symbols is not None and hasattr(state, "__len__") and len(state) == len(symbols):
                actions, _ = self.model.predict(state, deterministic=True)
                signals: list[StrategySignal] = []
                for sym, act in zip(symbols, actions, strict=False):
                    side = {0: "hold", 1: "buy", 2: "sell"}.get(int(act), "hold")
                    signals.append(
                        StrategySignal(symbol=sym, side=side, confidence=1.0, strategy="rl")
                    )
                return signals
            action, _ = self.model.predict(state, deterministic=True)
            side = {0: "hold", 1: "buy", 2: "sell"}.get(int(action), "hold")
            return StrategySignal(symbol="RL", side=side, confidence=1.0, strategy="rl")
        except (KeyError, ValueError, TypeError) as exc:
            logger.error("RL prediction failed: %s", exc)
            return None


class RLTrader(RLAgent):
    """Backward-compatible alias used by bot_engine."""
    pass


__all__ = ["RLAgent", "RLTrader", "is_rl_available", "PPO", "DummyVecEnv"]
