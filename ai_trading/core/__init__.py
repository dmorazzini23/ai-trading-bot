"""
Lightweight core package initializer.
Intentionally avoids importing .bot_engine to prevent circular imports and import-time side effects.
Import bot engine symbols directly from ai_trading.core.bot_engine where needed.
"""
import importlib
from .constants import TRADING_CONSTANTS
from .enums import AssetClass, OrderSide, OrderStatus, OrderType, RiskLevel, TimeFrame
__all__ = ['OrderSide', 'OrderType', 'OrderStatus', 'RiskLevel', 'TimeFrame', 'AssetClass', 'TRADING_CONSTANTS']

def __getattr__(name: str):
    if name == 'bot_engine':
        return importlib.import_module('.bot_engine', __name__)
    raise AttributeError(name)