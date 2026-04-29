"""Reinforcement learning trading utilities with optional dependencies."""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

import importlib
import inspect
import json
import sys
import time
import uuid
import zipfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, TYPE_CHECKING

from ai_trading.logging import get_logger

logger = get_logger(__name__)

# Exposed for tests to monkeypatch
PPO: Any | None = None
A2C: Any | None = None
DQN: Any | None = None
SAC: Any | None = None
TD3: Any | None = None
DummyVecEnv: Any | None = None

_SUPPORTED_SB3_ALGOS = ("PPO", "A2C", "DQN", "SAC", "TD3")

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
    except AI_TRADING_FALLBACK_EXCEPTIONS as exc:
        logger.exception("RL stack unavailable: %s", exc)
        return None
    global PPO, A2C, DQN, SAC, TD3, DummyVecEnv
    try:
        PPO = sb3.PPO
        A2C = sb3.A2C
        DQN = sb3.DQN
        SAC = sb3.SAC
        TD3 = sb3.TD3
        DummyVecEnv = sb3.common.vec_env.DummyVecEnv
    except AttributeError as exc:  # pragma: no cover - sanity guard
        raise ImportError("stable-baselines3 missing supported algorithms or DummyVecEnv") from exc
    return {"sb3": sb3, "gym": gym}


def is_rl_available() -> bool:
    """Return True if the optional RL dependencies can be imported."""
    return _load_rl_stack() is not None


class RLAgent:
    """Wrapper around a PPO policy for trading inference."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.model: Any | None = None

    @staticmethod
    def _algorithm_from_payload(payload: Any) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        for key in (
            "algorithm",
            "algo",
            "algo_name",
            "algorithm_name",
            "sb3_algorithm",
            "model_algorithm",
            "model_class",
            "algo_class",
            "class_name",
        ):
            value = payload.get(key)
            if isinstance(value, str):
                normalized = value.rsplit(".", maxsplit=1)[-1].strip().upper()
                if normalized in _SUPPORTED_SB3_ALGOS:
                    return normalized
        for key in ("metadata", "model_metadata", "training", "training_results"):
            found = RLAgent._algorithm_from_payload(payload.get(key))
            if found is not None:
                return found
        return None

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _algorithm_from_zip(model_path: Path) -> str | None:
        try:
            with zipfile.ZipFile(model_path) as model_zip:
                for member_name in (
                    "rl_model_metadata.json",
                    "metadata.json",
                    "meta.json",
                    "training_results.json",
                    "data",
                ):
                    try:
                        raw = model_zip.read(member_name)
                    except KeyError:
                        continue
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                        continue
                    found = RLAgent._algorithm_from_payload(payload)
                    if found is not None:
                        return found
        except (OSError, zipfile.BadZipFile):
            return None
        return None

    @classmethod
    def _resolve_algorithm(cls, model_path: Path) -> str:
        found = cls._algorithm_from_zip(model_path)
        if found is not None:
            return found
        candidates = [
            Path(f"{model_path}.meta.json"),
            model_path.with_suffix(f"{model_path.suffix}.meta.json")
            if model_path.suffix
            else Path(f"{model_path}.meta.json"),
            model_path.parent / "meta.json",
            model_path.parent / "training_results.json",
        ]
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if not candidate.is_file():
                continue
            found = cls._algorithm_from_payload(cls._read_json(candidate))
            if found is not None:
                return found
        filename = model_path.name.upper()
        for algorithm in _SUPPORTED_SB3_ALGOS:
            if f"_{algorithm}" in filename or f"-{algorithm}" in filename:
                return algorithm
        return "PPO"

    @staticmethod
    def _algorithm_class(algorithm: str) -> Any:
        algo_cls = globals().get(algorithm)
        if algo_cls is None:
            supported = ", ".join(_SUPPORTED_SB3_ALGOS)
            raise ImportError(
                f"RL algorithm {algorithm} unavailable; supported algorithms: {supported}"
            )
        return algo_cls

    def load(self) -> None:
        model_path = Path(self.model_path)
        if not is_rl_available():
            raise ImportError(
                "RL stack not available; install stable-baselines3, gymnasium, and torch"
            )
        if not model_path.exists():
            raise FileNotFoundError(f"RL model not found at {self.model_path}")
        algorithm = self._resolve_algorithm(model_path)
        algo_cls = self._algorithm_class(algorithm)
        try:
            self.model = algo_cls.load(self.model_path)
        except AI_TRADING_FALLBACK_EXCEPTIONS as exc:  # pragma: no cover - defensive guard
            logger.error(
                "RL model load failed for %s with %s: %s",
                self.model_path,
                algorithm,
                exc,
            )
            raise RuntimeError(f"RL model load failed for {self.model_path}") from exc

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
            state_len: int | None = None
            if symbols is not None:
                length_getter = getattr(state, "__len__", None)
                try:
                    state_len = int(length_getter()) if callable(length_getter) else None
                except (TypeError, ValueError):
                    state_len = None
            if symbols is not None and state_len == len(symbols):
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
    "A2C",
    "DQN",
    "SAC",
    "TD3",
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
        module = importlib.import_module(module_name)
    if module is None or not inspect.ismodule(module):
        raise AttributeError(f"module {__name__!r} has no attribute 'train'")
    sys.modules[module_name] = module
    globals()["train"] = module
    _TRAIN_MODULE_STATE["load_count"] += 1
    _TRAIN_MODULE_STATE["last_loaded_at"] = time.time()
    _TRAIN_MODULE_STATE["last_load_duration"] = time.perf_counter() - start
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
except AI_TRADING_FALLBACK_EXCEPTIONS:  # pragma: no cover - optional dependency missing or other import failure
    pass
