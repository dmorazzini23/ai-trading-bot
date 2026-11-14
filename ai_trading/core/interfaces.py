"""Clean architecture interfaces for dependency injection and separation of concerns.

Provides abstract interfaces for all major components to enable
proper dependency injection, testing, and modular design.

AI-AGENT-REF: Clean architecture interfaces for production trading platform
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from ai_trading.order.types import OrderSide

class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'

class OrderStatus(Enum):
    PENDING = 'pending'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELED = 'canceled'
    REJECTED = 'rejected'


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    timestamp: datetime
    metadata: dict[str, Any] = None

@dataclass
class Order:
    """Order data structure."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    quantity: float
    filled_quantity: float
    price: float | None
    filled_price: float | None
    timestamp: datetime
    metadata: dict[str, Any] = None

@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    metadata: dict[str, Any] = None


class IDependencyContainer(ABC):
    """Interface for dependency injection container."""

    @abstractmethod
    def register(self, interface: type, implementation: type, singleton: bool=False) -> None:
        """Register implementation for interface."""

    @abstractmethod
    def register_instance(self, interface: type, instance: Any) -> None:
        """Register instance for interface."""

    @abstractmethod
    def resolve(self, interface: type) -> Any:
        """Resolve implementation for interface."""

    @abstractmethod
    def configure(self, config: dict[str, Any]) -> None:
        """Configure container with configuration."""

_container: IDependencyContainer | None = None

