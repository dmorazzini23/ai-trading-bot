"""
Lightweight core package initializer.
Intentionally avoids importing .bot_engine to prevent circular imports and import-time side effects.
Import bot engine symbols directly from ai_trading.core.bot_engine where needed.
"""
import importlib
from .constants import TRADING_CONSTANTS
from .enums import AssetClass, OrderSide, OrderStatus, OrderType, RiskLevel, TimeFrame

__all__ = [
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'RiskLevel',
    'TimeFrame',
    'AssetClass',
    'TRADING_CONSTANTS',
]


def _load_bot_engine():
    return importlib.import_module('.bot_engine', __name__)


_LOOKUP = {'bot_engine': _load_bot_engine}


def __getattr__(name: str):
    factory = _LOOKUP.get(name)
    if factory is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = factory()
    globals()[name] = value
    return value
