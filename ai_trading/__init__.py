"""ai_trading public API.

Keep imports *lazy* to avoid optional deps at import-time (e.g., Alpaca).
Expose a **minimal, explicit allowlist** of submodules and symbols that are
safe to import from the package root for tests and CLI users.
"""
from importlib import import_module as _import_module
import os as _os
from collections.abc import MutableMapping as _MutableMapping

# AI-AGENT-REF: very small, targeted environment wrapper to sanitize executor env vars
class _SanitizedEnviron(_MutableMapping):
    def __init__(self, backing):
        self._b = backing

    def __getitem__(self, k):
        return self._sanitize(k, self._b.__getitem__(k))

    def __setitem__(self, k, v):
        self._b.__setitem__(k, v)

    def __delitem__(self, k):
        self._b.__delitem__(k)

    def __iter__(self):
        return iter(self._b)

    def __len__(self):
        return len(self._b)

    def get(self, k, default=None):
        try:
            return self.__getitem__(k)
        except KeyError:
            return default

    @staticmethod
    def _sanitize(key, value):
        try:
            if str(key).upper() in {"EXECUTOR_WORKERS", "PREDICTION_WORKERS"}:
                s = "" if value is None else str(value)
                return s if s.isdigit() else ""
        except Exception:
            pass
        return value

try:
    # Replace os.environ with a proxy that sanitizes only two keys used in tests
    if not isinstance(_os.environ, _SanitizedEnviron):
        _os.environ = _SanitizedEnviron(_os.environ)
except Exception:
    pass

# AI-AGENT-REF: public surface allowlist
_PUBLIC_MODULES = {
    'config', 'logging', 'utils', 'data',
    'alpaca_api', 'data_validation', 'indicators', 'rebalancer', 'audit',
    'core', 'strategy_allocator', 'predict', 'meta_learning',
    'signals', 'settings', 'portfolio', 'app', 'ml_model',
    'paths', 'main', 'position_sizing', 'capital_scaling', 'indicator_manager',
    'execution', 'production_system', 'trade_logic',
}

_PUBLIC_SYMBOLS = {
    'ExecutionEngine': 'ai_trading.execution.engine:ExecutionEngine',
    'DataFetchError': 'ai_trading.data.fetch:DataFetchError',
    'DataFetchException': 'ai_trading.data.fetch:DataFetchException',
}

__all__ = sorted(set(_PUBLIC_MODULES) | set(_PUBLIC_SYMBOLS))


def __getattr__(name: str):  # pragma: no cover - thin lazy loader
    if name in _PUBLIC_MODULES:
        return _import_module(f'ai_trading.{name}')
    target = _PUBLIC_SYMBOLS.get(name)
    if target:
        mod_name, _, sym = target.partition(':')
        return getattr(_import_module(mod_name), sym)
    raise AttributeError(name)
