"""Helpers for lazily loading heavy third-party modules."""

from __future__ import annotations

import importlib
from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType

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
    """Return a proxy for :mod:`pandas` loaded on first use."""
    return _LazyModule("pandas")

@lru_cache(maxsize=None)
def load_pandas_market_calendars() -> ModuleType | None:
    """Return a proxy for :mod:`pandas_market_calendars` if available."""
    if find_spec("pandas_market_calendars") is None:
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
