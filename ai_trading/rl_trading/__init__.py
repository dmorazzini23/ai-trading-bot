"""Reinforcement learning trading utilities with optional dependencies."""
from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ai_trading.logging import get_logger

logger = get_logger(__name__)

# Exposed for tests to monkeypatch
PPO: Any | None = None
DummyVecEnv: Any | None = None

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from ai_trading.strategies.base import StrategySignal  # noqa: F401
    from . import train as _train_module

    train = _train_module


@lru_cache(maxsize=1)
def _load_rl_stack() -> dict[str, Any] | None:
    """Attempt to import the optional RL stack and cache the result."""
    try:
        sb3 = importlib.import_module("stable_baselines3")
        gym = importlib.import_module("gymnasium")
        importlib.import_module("torch")
    except Exception as exc:
        logger.exception("RL stack unavailable: %s", exc)
        return None
    global PPO, DummyVecEnv
    try:
        PPO = sb3.PPO
        DummyVecEnv = sb3.common.vec_env.DummyVecEnv
    except AttributeError as exc:  # pragma: no cover - sanity guard
        raise ImportError("stable-baselines3 missing PPO or DummyVecEnv") from exc
    return {"sb3": sb3, "gym": gym}


def is_rl_available() -> bool:
    """Return True if the optional RL dependencies can be imported."""
    return _load_rl_stack() is not None


class RLAgent:
    """Wrapper around a PPO policy for trading inference."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.model: Any | None = None

    def _load_stub_model(self, model_path: Path) -> None:
        """Load the lightweight stub model used when RL dependencies are missing."""

        from . import train  # imported lazily to avoid optional deps at import time

        logger.warning(
            "RL stack unavailable – falling back to stub model for %s",
            model_path,
        )
        if model_path.exists():
            try:
                self.model = train.Model.load(model_path)
                return
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "Failed to load stub RL model from %s: %s; creating fresh stub",
                    model_path,
                    exc,
                )
        self.model = train.Model(train.TrainingConfig(model_path=str(model_path)))

    def load(self) -> None:
        model_path = Path(self.model_path)
        rl_ready = is_rl_available()
        is_stub = getattr(PPO, "__name__", "") == "_SB3Stub"
        if not rl_ready or PPO is None or is_stub:
            self._load_stub_model(model_path)
            return
        if model_path.exists():
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


__all__ = [
    "DummyVecEnv",
    "PPO",
    "RLAgent",
    "RLTrader",
    "is_rl_available",
    "train",
]


def _load_train_module() -> Any:
    """Dynamically import :mod:`ai_trading.rl_trading.train` when requested."""

    module_name = f"{__name__}.train"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(
            f"module {__name__!r} has no attribute 'train'"
        ) from exc
    sys.modules.setdefault("ai_trading.rl_trading.train", module)
    globals()["train"] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy loader
    if name == "train":
        return _load_train_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - keep introspection predictable
    return sorted({*globals(), "train"})


try:  # Eagerly import to keep a stable module reference for reloads.
    _load_train_module()
except AttributeError:  # pragma: no cover - optional dependency missing in tests
    train = None
