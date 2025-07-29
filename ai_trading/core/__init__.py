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
]