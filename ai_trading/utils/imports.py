import importlib
import importlib.util
import logging

_log = logging.getLogger(__name__)


def _try_import(module_name: str, cls_name: str):
    """find_spec + import_module; returns class or None; logs exceptions."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, cls_name, None)
    except Exception as e:
        _log.error("Failed to import %s (%s): %s", module_name, cls_name, e)
        return None


def resolve_risk_engine_cls():
    cls = _try_import("ai_trading.risk_engine", "RiskEngine")
    if cls: 
        return cls
    cls = _try_import("scripts.risk_engine", "RiskEngine")
    if cls: 
        return cls
    return None


def resolve_strategy_allocator_cls():
    cls = _try_import("ai_trading.strategy_allocator", "StrategyAllocator")
    if cls: 
        return cls
    cls = _try_import("scripts.strategy_allocator", "StrategyAllocator")
    if cls: 
        return cls
    return None