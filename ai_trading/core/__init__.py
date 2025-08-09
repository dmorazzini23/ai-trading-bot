"""
Core trading module for institutional-grade trading platform.

This module provides fundamental trading enums, constants, and core infrastructure
components used throughout the AI trading platform. It includes standardized
enumerations for order management, risk levels, timeframes, and consolidated
trading constants for system-wide configuration.

Exports:
    - OrderSide: Buy/sell enumeration for order direction
    - OrderType: Market, limit, stop order types
    - OrderStatus: Order execution state tracking
    - RiskLevel: Conservative, moderate, aggressive risk levels
    - TimeFrame: Market data timeframe definitions
    - AssetClass: Asset classification for portfolio diversification
    - TRADING_CONSTANTS: Consolidated trading configuration parameters
    - BotState: Core bot state management
    - pre_trade_health_check: Pre-trade health validation
    - run_all_trades_worker: Main trading worker function
"""

# Import core enums
from .enums import (
    OrderSide,
    OrderType, 
    OrderStatus,
    RiskLevel,
    TimeFrame,
    AssetClass
)

# Import trading constants
from .constants import TRADING_CONSTANTS

# Import bot engine components  
# AI-AGENT-REF: Use lazy import to prevent config crash at import time
def __getattr__(name):
    """Lazy import bot_engine components to prevent import-time config crashes."""
    if name in ("BotState", "pre_trade_health_check", "run_all_trades_worker"):
        from .bot_engine import BotState, pre_trade_health_check, run_all_trades_worker
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Define explicit exports
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
    
    # Bot engine components
    "BotState",
    "pre_trade_health_check", 
    "run_all_trades_worker",
]