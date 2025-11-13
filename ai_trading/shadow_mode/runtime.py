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
        self._module._load()
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
                # Temporarily remove the placeholder to resolve the real spec/loader,
                # otherwise ``find_spec`` may return our own lazy spec causing recursion.
                removed_placeholder = False
                if sys.modules.get(self.__name__) is self:
                    sys.modules.pop(self.__name__, None)
                    removed_placeholder = True

                try:
                    spec = importlib.util.find_spec(self.__name__)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Unable to locate spec for {self.__name__!r}")
                    module = importlib.util.module_from_spec(spec)
                    # Ensure relative imports inside the real module can resolve
                    sys.modules[self.__name__] = module
                    try:
                        spec.loader.exec_module(module)
                    finally:
                        # Restore the placeholder in sys.modules for future imports
                        sys.modules[self.__name__] = self
                    self._real = module
                finally:
                    if not removed_placeholder and sys.modules.get(self.__name__) is not self:
                        # Defensive: keep placeholder registered
                        sys.modules[self.__name__] = self
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
        self._mirror_real_attributes()
        return self._real

    def _mirror_real_attributes(self) -> None:
        real = self._real
        if real is None:
            return
        skip = {"_real", "_loading", "__loader__", "__spec__", "__dict__"}
        for key, value in real.__dict__.items():
            if key in skip:
                continue
            setattr(self, key, value)

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
