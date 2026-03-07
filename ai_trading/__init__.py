"""ai_trading public API.

Exports are resolved lazily to keep package import side-effect free.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any

PYTEST_DONT_REWRITE = ["ai_trading"]

# AI-AGENT-REF: public surface allowlist
_EXPORTS = {
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
    "predict": "ai_trading.predict",
    "production_system": "ai_trading.production_system",
    "rebalancer": "ai_trading.rebalancer",
    "settings": "ai_trading.settings",
    "signals": "ai_trading.signals",
    "strategy_allocator": "ai_trading.strategy_allocator",
    "trade_logic": "ai_trading.trade_logic",
    "utils": "ai_trading.utils",
    "ExecutionEngine": "ai_trading.execution.engine:ExecutionEngine",
    "DataFetchError": "ai_trading.data.fetch:DataFetchError",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, _, attr_name = target.partition(":")
    module_obj = _import_module(module_name)
    resolved = getattr(module_obj, attr_name) if attr_name else module_obj
    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
