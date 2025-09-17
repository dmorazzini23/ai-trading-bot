from __future__ import annotations

import importlib
import sys
from types import ModuleType


class _LazyModule(ModuleType):
    """Proxy module that loads the real module on first attribute access."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._real: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._real is None:
            # Remove the proxy so ``import_module`` loads the real module
            sys.modules.pop(self.__name__, None)
            self._real = importlib.import_module(self.__name__)
        # Ensure ``sys.modules`` contains the loaded module for future imports
        sys.modules[self.__name__] = self._real
        return self._real

    def __getattr__(self, item: str):  # pragma: no cover - exercised via tests
        return getattr(self._load(), item)


def ensure_alpaca_api() -> ModuleType:
    """Register a lazy loader for ``ai_trading.alpaca_api``.

    The actual module import is deferred until an attribute is accessed, but a
    placeholder is inserted into :mod:`sys.modules` so that subsequent imports
    resolve correctly.
    """

    name = "ai_trading.alpaca_api"
    mod = sys.modules.get(name)
    if isinstance(mod, _LazyModule):
        return mod

    lazy = _LazyModule(name)
    if mod is not None:
        lazy._real = mod
    sys.modules[name] = lazy
    return lazy


# Ensure placeholder is registered when this module is imported
ensure_alpaca_api()
