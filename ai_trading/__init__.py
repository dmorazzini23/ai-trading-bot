"""ai_trading public API.

Exports are resolved lazily to keep package import side-effect free.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any
import warnings

PYTEST_DONT_REWRITE = ["ai_trading"]

# AI-AGENT-REF: public surface allowlist
_PUBLIC_EXPORTS = {
    "alpaca_api": "ai_trading.alpaca_api",
    "app": "ai_trading.app",
    "audit": "ai_trading.audit",
    "capital_scaling": "ai_trading.capital_scaling",
    "config": "ai_trading.config",
    "core": "ai_trading.core",
    "data": "ai_trading.data",
    "data_validation": "ai_trading.data_validation",
    "execution": "ai_trading.execution",
    "indicator_manager": "ai_trading.indicator_manager",
    "indicators": "ai_trading.indicators",
    "logging": "ai_trading.logging",
    "main": "ai_trading.main",
    "meta_learning": "ai_trading.meta_learning",
    "ml_model": "ai_trading.ml_model",
    "paths": "ai_trading.paths",
    "portfolio": "ai_trading.portfolio",
    "position_sizing": "ai_trading.position_sizing",
    "rebalancer": "ai_trading.rebalancer",
    "settings": "ai_trading.settings",
    "signals": "ai_trading.signals",
    "strategy_allocator": "ai_trading.strategy_allocator",
    "utils": "ai_trading.utils",
    "ExecutionEngine": "ai_trading.execution:ExecutionEngine",
    "DataFetchError": "ai_trading.data.fetch:DataFetchError",
}

_LEGACY_RESEARCH_EXPORTS = {
    "predict": "ai_trading.predict",
    "trade_logic": "ai_trading.trade_logic",
}

_EXPORTS = {**_PUBLIC_EXPORTS, **_LEGACY_RESEARCH_EXPORTS}

__all__ = sorted(_PUBLIC_EXPORTS)

_DEPRECATED_RESEARCH_EXPORTS = {
    "predict": "ai_trading.predict is deprecated as a package-level live API; use it only for research utilities.",
    "trade_logic": "ai_trading.trade_logic is deprecated as a package-level live API; use it only for research utilities.",
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warning_message = _DEPRECATED_RESEARCH_EXPORTS.get(name)
    if warning_message is not None:
        warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
    if name == "ExecutionEngine":
        execution_module = _import_module("ai_trading.execution")
        return execution_module.select_execution_engine()
    module_name, _, attr_name = target.partition(":")
    module_obj = _import_module(module_name)
    resolved = getattr(module_obj, attr_name) if attr_name else module_obj
    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
