from __future__ import annotations

import importlib
import sys
from types import ModuleType


class _LazyModule(ModuleType):
    """Proxy module that loads the real module on first attribute access."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._real: ModuleType | None = None
        self._loading = False

    def _load(self) -> ModuleType:
        if self._real is None and not self._loading:
            self._loading = True
            try:
                # Remove the proxy so ``import_module`` loads the real module
                sys.modules.pop(self.__name__, None)
                module = importlib.import_module(self.__name__)
                self._real = module
            finally:
                # Keep the proxy registered for future imports, even on failure.
                sys.modules[self.__name__] = self
                self._loading = False
        elif self._real is None and self._loading:
            # Re-entrant access while loading should behave as if the module is
            # still the proxy to avoid recursion issues.
            sys.modules[self.__name__] = self
        else:
            sys.modules[self.__name__] = self

        if self._real is None:
            raise ImportError(f"Failed to load module {self.__name__!r}")
        return self._real

    def __getattr__(self, item: str):  # pragma: no cover - exercised via tests
        return getattr(self._load(), item)

    def submit_order(self, *args, **kwargs):
        """Proxy order submission to the real Alpaca module without reloading twice."""

        module = self._load()
        submit = getattr(module, "submit_order")
        return submit(*args, **kwargs)


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
