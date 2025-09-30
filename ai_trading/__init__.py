"""ai_trading public API.

Keep imports *lazy* to avoid optional deps at import-time (e.g., Alpaca).
Expose a **minimal, explicit allowlist** of submodules and symbols that are
safe to import from the package root for tests and CLI users.
"""
import os
import sys
from pathlib import Path
from importlib import import_module as _import_module

# Ensure a prior ``python -m ai_trading`` run does not leave behind a stale
# ``ai_trading.__main__`` module entry that would short-circuit lazy exports
# during package initialization.
sys.modules.pop(f"{__name__}.__main__", None)

PYTEST_DONT_REWRITE = ["ai_trading"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_root_str = str(_REPO_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
_existing_pythonpath = os.environ.get("PYTHONPATH")
if not _existing_pythonpath:
    os.environ["PYTHONPATH"] = _root_str
elif _root_str not in _existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{_root_str}{os.pathsep}{_existing_pythonpath}"

# AI-AGENT-REF: public surface allowlist
_EXPORTS = {
    'alpaca_api': 'ai_trading.alpaca_api',
    'app': 'ai_trading.app',
    'audit': 'ai_trading.audit',
    'capital_scaling': 'ai_trading.capital_scaling',
    'config': 'ai_trading.config',
    'core': 'ai_trading.core',
    'data': 'ai_trading.data',
    'data_validation': 'ai_trading.data_validation',
    'execution': 'ai_trading.execution',
    'indicator_manager': 'ai_trading.indicator_manager',
    'indicators': 'ai_trading.indicators',
    'logging': 'ai_trading.logging',
    'main': 'ai_trading.main',
    'meta_learning': 'ai_trading.meta_learning',
    'ml_model': 'ai_trading.ml_model',
    'paths': 'ai_trading.paths',
    'portfolio': 'ai_trading.portfolio',
    'position_sizing': 'ai_trading.position_sizing',
    'predict': 'ai_trading.predict',
    'production_system': 'ai_trading.production_system',
    'rebalancer': 'ai_trading.rebalancer',
    'settings': 'ai_trading.settings',
    'signals': 'ai_trading.signals',
    'strategy_allocator': 'ai_trading.strategy_allocator',
    'trade_logic': 'ai_trading.trade_logic',
    'utils': 'ai_trading.utils',
    'ExecutionEngine': 'ai_trading.execution.engine:ExecutionEngine',
    'DataFetchError': 'ai_trading.data.fetch:DataFetchError',
    'DataFetchException': 'ai_trading.data.fetch:DataFetchException',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):  # pragma: no cover - thin lazy loader
    target = _EXPORTS.get(name)
    if not target:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, _, attr = target.partition(':')
    mod = _import_module(mod_name)
    obj = getattr(mod, attr) if attr else mod
    globals()[name] = obj
    return obj
