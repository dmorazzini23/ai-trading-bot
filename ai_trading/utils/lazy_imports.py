"""Helpers for lazily loading heavy third-party modules."""

from __future__ import annotations

import importlib
from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType
import sys

from ai_trading.logging import get_logger

class _LazyModule(ModuleType):
    """Proxy module that loads the real module upon first attribute access."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return self._module

    def __getattr__(self, item: str):  # pragma: no cover - passthrough
        return getattr(self._load(), item)

@lru_cache(maxsize=None)
def load_pandas() -> ModuleType:
    """Return the canonical :mod:`pandas` module, importing eagerly once."""

    module = sys.modules.get("pandas")
    if module is not None and getattr(module, "DataFrame", None) is not None:
        return module
    try:
        module = importlib.import_module("pandas")
    except ModuleNotFoundError:
        if module is not None:
            return module
        raise
    sys.modules["pandas"] = module
    return module

@lru_cache(maxsize=None)
def load_pandas_market_calendars() -> ModuleType | None:
    """Return a proxy for :mod:`pandas_market_calendars` if available."""
    try:
        spec = find_spec("pandas_market_calendars")
    except ValueError:
        # Some test harnesses insert a stub with __spec__ None; treat as unavailable
        return None
    if spec is None:
        return None
    return _LazyModule("pandas_market_calendars")


@lru_cache(maxsize=None)
def load_pandas_ta() -> ModuleType | None:
    """Return :mod:`pandas_ta` if available, logging once when missing."""
    try:
        return importlib.import_module("pandas_ta")
    except Exception:  # pragma: no cover - optional dependency
        get_logger(__name__).info(
            "PANDAS_TA_MISSING", extra={"hint": "pip install pandas-ta"}
        )
        return None


@lru_cache(maxsize=None)
def _load_sklearn_submodule(name: str) -> ModuleType | None:
    """Return a proxy for a :mod:`sklearn` submodule if available."""
    mod_name = f"sklearn.{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if find_spec(mod_name) is None:
        return None
    return _LazyModule(mod_name)


def load_sklearn_linear_model() -> ModuleType | None:
    """Return :mod:`sklearn.linear_model` lazily."""
    return _load_sklearn_submodule("linear_model")


def load_sklearn_pipeline() -> ModuleType | None:
    """Return :mod:`sklearn.pipeline` lazily."""
    return _load_sklearn_submodule("pipeline")


def load_sklearn_preprocessing() -> ModuleType | None:
    """Return :mod:`sklearn.preprocessing` lazily."""
    return _load_sklearn_submodule("preprocessing")


def load_sklearn_model_selection() -> ModuleType | None:
    """Return :mod:`sklearn.model_selection` lazily."""
    return _load_sklearn_submodule("model_selection")


def load_sklearn_ensemble() -> ModuleType | None:
    """Return :mod:`sklearn.ensemble` lazily."""
    return _load_sklearn_submodule("ensemble")


def load_sklearn_metrics() -> ModuleType | None:
    """Return :mod:`sklearn.metrics` lazily."""
    return _load_sklearn_submodule("metrics")
