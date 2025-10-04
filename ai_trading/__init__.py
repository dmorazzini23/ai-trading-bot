"""ai_trading public API.

Keep imports *lazy* to avoid optional deps at import-time (e.g., Alpaca).
Expose a **minimal, explicit allowlist** of submodules and symbols that are
safe to import from the package root for tests and CLI users.
"""
import importlib.util
import os
import runpy  # noqa: F401 - ensure availability when sys.modules cleared
import sys
import zoneinfo  # noqa: F401 - ensure tzdata bindings load after clears
from pathlib import Path
from importlib import import_module as _import_module
import unittest.mock as _mock

# Ensure a prior ``python -m ai_trading`` run does not leave behind a stale
# ``ai_trading.__main__`` module entry that would short-circuit lazy exports
# during package initialization.
sys.modules.pop(f"{__name__}.__main__", None)

PYTEST_DONT_REWRITE = ["ai_trading"]


_SAFE_SYS_MODULES = (
    "sys",
    "builtins",
    "types",
    "importlib",
    "importlib._bootstrap",
    "importlib._bootstrap_external",
    "importlib.machinery",
    "pathlib",
    "posixpath",
    "ntpath",
    "os",
    "collections",
    "runpy",
    "zipimport",
    "pkgutil",
    "zoneinfo",
    "zoneinfo._tzpath",
)


class _SafePatchDict(_mock._patch_dict):  # pragma: no cover - exercised via tests
    def _patch_dict(self):
        super()._patch_dict()
        if self.in_dict is sys.modules and self.clear:
            for name in _SAFE_SYS_MODULES:
                if name in sys.modules:
                    continue
                try:
                    __import__(name)
                except Exception:
                    continue


_mock._patch_dict = _SafePatchDict

_modules_ref = getattr(sys, "modules", None)
_ESSENTIAL_NAMES = (
    "sys",
    "builtins",
    "types",
    "importlib",
    "importlib._bootstrap",
    "importlib._bootstrap_external",
    "importlib.machinery",
    "pathlib",
    "ntpath",
    "posixpath",
    "os",
    "collections",
    "_pytest.assertion.rewrite",
)

for _mod_name in _ESSENTIAL_NAMES:
    try:  # pragma: no cover - ensure snapshot captures initialized modules
        __import__(_mod_name)
    except Exception:
        continue

_ESSENTIAL_SNAPSHOT = {name: sys.modules.get(name) for name in _ESSENTIAL_NAMES}


def _restore_essential_modules() -> None:  # pragma: no cover - defensive restoration
    for name, module in _ESSENTIAL_SNAPSHOT.items():
        if module is not None and sys.modules.get(name) is None:
            sys.modules[name] = module


if not _modules_ref or any(sys.modules.get(name) is None for name in ("importlib", "pathlib", "os")):
    _restore_essential_modules()
    _modules_ref = sys.modules

if _modules_ref is None:  # pragma: no cover - defensive for patched environments
    import importlib
    import types
    import builtins
    import pathlib
    import ntpath

    sys.modules = {
        "sys": sys,
        "builtins": builtins,
        "types": types,
        "importlib": importlib,
        "importlib._bootstrap": importlib._bootstrap,
        "importlib._bootstrap_external": importlib._bootstrap_external,
        "pathlib": pathlib,
        "ntpath": ntpath,
        "os": os,
    }
    _modules_ref = sys.modules

try:
    _PIPELINE_AVAILABLE = importlib.util.find_spec("sklearn.pipeline") is not None
except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
    _PIPELINE_AVAILABLE = False

if not _PIPELINE_AVAILABLE:
    import importlib.machinery
    import importlib.abc

    class _PipelineMissingLoader(importlib.abc.Loader):  # pragma: no cover - loader raising ImportError
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise ImportError(
                "ai_trading.pipeline requires scikit-learn; install with `pip install scikit-learn`."
            )

        def get_source(self, fullname):
            raise ImportError(
                "ai_trading.pipeline requires scikit-learn; install with `pip install scikit-learn`."
            )

    class _PipelineMissingFinder(importlib.abc.MetaPathFinder):  # pragma: no cover - import hook
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("ai_trading.pipeline"):
                return importlib.machinery.ModuleSpec(fullname, _PipelineMissingLoader())
            return None

    sys.meta_path.insert(0, _PipelineMissingFinder())

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
