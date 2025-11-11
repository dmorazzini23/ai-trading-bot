from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType


class _LazyLoader:
    """Minimal loader that keeps the placeholder stable for ``importlib.reload``."""

    def __init__(self, module: "_LazyModule") -> None:
        self._module = module

    def create_module(self, spec: ModuleSpec) -> ModuleType:  # pragma: no cover - thin shim
        del spec
        return self._module

    def exec_module(self, module: ModuleType) -> None:  # pragma: no cover - thin shim
        del module
        return None


class _LazyModule(ModuleType):
    """Proxy module that loads the real module on first attribute access."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._real: ModuleType | None = None
        self._loading = False
        loader = _LazyLoader(self)
        self.__loader__ = loader
        self.__spec__ = ModuleSpec(name, loader)

    def _load(self) -> ModuleType:
        if self._real is None and not self._loading:
            self._loading = True
            try:
                spec = importlib.util.find_spec(self.__name__)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Unable to locate spec for {self.__name__!r}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[self.__name__] = module
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.modules[self.__name__] = self
                self._real = module
            finally:
                self._loading = False
        elif self._real is None and self._loading:
            # Re-entrant access while loading should behave as if the module is
            # still the proxy to avoid recursion issues.
            sys.modules[self.__name__] = self
        else:
            sys.modules[self.__name__] = self

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
