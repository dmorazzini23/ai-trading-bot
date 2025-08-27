import importlib
import importlib.util
from ai_trading.logging import get_logger
logger = get_logger(__name__)

def _try_import(module_name: str, cls_name: str):
    """find_spec + import_module; returns class or None; logs exceptions."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, cls_name, None)
    except (KeyError, ValueError, TypeError) as e:
        logger.error('Failed to import %s (%s): %s', module_name, cls_name, e)
        return None

def resolve_risk_engine_cls():
    cls = _try_import('ai_trading.risk.engine', 'RiskEngine')
    if cls:
        return cls
    return None

def resolve_strategy_allocator_cls():
    cls = _try_import('ai_trading.strategy_allocator', 'StrategyAllocator')
    if cls:
        return cls
    cls = _try_import('ai_trading.strategies.performance_allocator', 'PerformanceBasedAllocator')
    if cls:
        return cls
    return None
