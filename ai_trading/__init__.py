"""ai_trading public API.

Keep imports *lazy* to avoid optional deps at import-time (e.g., Alpaca).
Expose a **minimal, explicit allowlist** of submodules and symbols that are
safe to import from the package root for tests and CLI users.
"""
from importlib import import_module as _import_module
import os as _os

# AI-AGENT-REF: sanitize specific executor env vars once at import time (no wrapper)
try:
    for _k in ("EXECUTOR_WORKERS", "PREDICTION_WORKERS"):
        _v = _os.environ.get(_k)
        if _v is not None and not str(_v).isdigit():
            _os.environ[_k] = ""
    _orig_getenv = _os.getenv
    def _sanitized_getenv(key, default=None):  # type: ignore[override]
        if str(key).upper() in {"EXECUTOR_WORKERS", "PREDICTION_WORKERS"}:
            val = _orig_getenv(key, default)
            try:
                return val if (val is None or str(val).isdigit()) else ""
            except Exception:
                return ""
        return _orig_getenv(key, default)
    _os.getenv = _sanitized_getenv  # type: ignore[assignment]
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
