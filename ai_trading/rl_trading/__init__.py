"""Reinforcement learning trading utilities with optional dependencies."""
from __future__ import annotations

import importlib
import importlib.util
import importlib.machinery
import inspect
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, TYPE_CHECKING, NoReturn

from ai_trading.logging import get_logger

logger = get_logger(__name__)

# Exposed for tests to monkeypatch
PPO: Any | None = None
DummyVecEnv: Any | None = None

_TRAIN_MODULE_STATE: dict[str, Any] = {
    "loader_id": uuid.uuid4().hex,
    "load_count": 0,
    "last_loaded_at": None,
    "last_load_duration": None,
}

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
        if not model_path.exists():
            logger.error("RL model not found at %s", self.model_path)
            self._load_stub_model(model_path)
            return
        try:
            self.model = PPO.load(self.model_path)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("RL model load failed for %s: %s", self.model_path, exc)
            self._load_stub_model(model_path)

    def predict(
        self, state: Iterable[Any] | Any, symbols: list[str] | None = None
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
                    bar_ts = None
                    try:
                        if isinstance(state, dict):
                            bar_ts = state.get("bar_ts")
                        elif hasattr(state, "bar_ts"):
                            bar_ts = getattr(state, "bar_ts")
                    except (TypeError, AttributeError):
                        bar_ts = None
                    signals.append(
                        StrategySignal(
                            symbol=sym,
                            side=side,
                            strength=1.0,
                            confidence=1.0,
                            strategy="rl",
                            metadata={"bar_ts": bar_ts},
                        )
                    )
                return signals
            action, _ = self.model.predict(state, deterministic=True)
            side = {0: "hold", 1: "buy", 2: "sell"}.get(int(action), "hold")
            bar_ts = None
            try:
                if isinstance(state, dict):
                    bar_ts = state.get("bar_ts")
                elif hasattr(state, "bar_ts"):
                    bar_ts = getattr(state, "bar_ts")
            except (TypeError, AttributeError):
                bar_ts = None
            return StrategySignal(
                symbol="RL",
                side=side,
                strength=1.0,
                confidence=1.0,
                strategy="rl",
                metadata={"bar_ts": bar_ts},
            )
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


def _load_train_module() -> ModuleType:
    """Dynamically import :mod:`ai_trading.rl_trading.train` when requested."""

    start = time.perf_counter()
    module_name = f"{__name__}.train"
    module = sys.modules.get(module_name)
    if module is not None and not inspect.ismodule(module):
        logger.warning(
            "Non-module entry for %s detected in sys.modules; re-importing", module_name
        )
        module = None
        sys.modules.pop(module_name, None)
    if module is None:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            module = _load_train_stub(module_name, exc)
        except ImportError as exc:  # pragma: no cover - defensive guard
            module = _load_train_stub(module_name, exc)
    if module is None or not inspect.ismodule(module):
        raise AttributeError(f"module {__name__!r} has no attribute 'train'")
    sys.modules[module_name] = module
    globals()["train"] = module
    _TRAIN_MODULE_STATE["load_count"] += 1
    _TRAIN_MODULE_STATE["last_loaded_at"] = time.time()
    _TRAIN_MODULE_STATE["last_load_duration"] = time.perf_counter() - start
    return module


def _load_train_stub(module_name: str, original_exc: Exception) -> ModuleType:
    """Load the lightweight stub implementation for the train module."""

    stub_name = f"{__name__}._train_stub"
    spec = importlib.util.find_spec(stub_name)
    if spec is None or spec.origin is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute 'train'"
        ) from original_exc
    loader = importlib.machinery.SourceFileLoader(module_name, spec.origin)
    stub_spec = importlib.util.spec_from_loader(module_name, loader, origin=spec.origin)
    if stub_spec is None or stub_spec.loader is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute 'train'"
        ) from original_exc
    module = importlib.util.module_from_spec(stub_spec)
    sys.modules[module_name] = module
    try:
        stub_spec.loader.exec_module(module)
    except Exception as stub_exc:  # pragma: no cover - defensive guard
        raise AttributeError(
            f"module {__name__!r} has no attribute 'train'"
        ) from stub_exc
    module.__dict__.setdefault("__fallback_exception__", original_exc)
    module.__dict__.setdefault("USING_RL_TRAIN_STUB", True)
    logger.warning(
        "Using RL training stub due to import failure: %s", original_exc
    )
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover
    """PEP 562 hook to keep ``train`` lazily importable and stable."""

    if name == "train":
        return _load_train_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - keep introspection predictable
    return sorted({*globals(), "train"})


try:  # Eagerly import to keep a stable module reference for reloads.
    train = _load_train_module()
except Exception as exc:  # pragma: no cover - optional dependency missing or other import failure
    stub = ModuleType(f"{__name__}.train")

    def _raise_import_error(*_a: Any, _exc: Exception = exc, **_k: Any) -> NoReturn:
        raise ImportError(
            "RL stack not available; install stable-baselines3, gymnasium, and torch"
        ) from _exc

    stub.__dict__.update(
        {
            "train": _raise_import_error,
            "USING_RL_TRAIN_STUB": True,
            "__fallback_exception__": exc,
        }
    )
    sys.modules.setdefault(f"{__name__}.train", stub)
    train = stub
