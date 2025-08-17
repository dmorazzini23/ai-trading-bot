"""
Lightweight core package initializer.
Intentionally avoids importing .bot_engine to prevent circular imports and import-time side effects.
Import bot engine symbols directly from ai_trading.core.bot_engine where needed.
"""

# Import core enums and constants (safe imports)
import importlib

from .constants import TRADING_CONSTANTS
from .enums import AssetClass, OrderSide, OrderStatus, OrderType, RiskLevel, TimeFrame

# Re-export only light, side-effect-free symbols:
__all__ = [
    # Order management enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    # Risk and strategy enums
    "RiskLevel",
    "TimeFrame",
    "AssetClass",
    # Configuration constants
    "TRADING_CONSTANTS",
    # For everything else, consumers must do:
    #   from ai_trading.core.bot_engine import BotState, run_all_trades_worker
]


def __getattr__(name: str):  # AI-AGENT-REF: lazy bot_engine access
    if name == "bot_engine":
        return importlib.import_module(".bot_engine", __name__)
    raise AttributeError(name)
