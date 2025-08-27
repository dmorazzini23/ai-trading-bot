"""ai_trading public API.

Keep imports *lazy* to avoid optional deps at import-time (e.g., Alpaca).
Expose a **minimal, explicit allowlist** of submodules and symbols that are
safe to import from the package root for tests and CLI users.
"""
from importlib import import_module as _import_module

# AI-AGENT-REF: public surface allowlist
_PUBLIC_MODULES = {
    'config', 'logging', 'utils',
    'alpaca_api', 'data_validation', 'indicators', 'rebalancer', 'audit',
    'core', 'strategy_allocator', 'predict', 'meta_learning',
    'signals', 'settings', 'runner', 'portfolio', 'app', 'ml_model',
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

