"""Clean architecture interfaces for dependency injection and separation of concerns.

Provides abstract interfaces for all major components to enable
proper dependency injection, testing, and modular design.

AI-AGENT-REF: Clean architecture interfaces for production trading platform
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd

class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

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
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    side: OrderSide
    confidence: float
    timestamp: datetime
    price: float | None = None
    quantity: int | None = None
    metadata: dict[str, Any] = None

@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: int
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
    quantity: int
    filled_quantity: int
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

class IDataProvider(ABC):
    """Interface for market data providers."""

    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str='1min', start: datetime | None=None, end: datetime | None=None) -> pd.DataFrame:
        """Get historical market data."""

    @abstractmethod
    async def get_real_time_data(self, symbol: str) -> AsyncIterator[MarketData]:
        """Get real-time market data stream."""

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""

    @abstractmethod
    async def get_market_status(self) -> dict[str, Any]:
        """Get market status information."""

class IFundamentalDataProvider(ABC):
    """Interface for fundamental data providers."""

    @abstractmethod
    async def get_company_info(self, symbol: str) -> dict[str, Any]:
        """Get company fundamental information."""

    @abstractmethod
    async def get_financial_statements(self, symbol: str) -> dict[str, Any]:
        """Get financial statements."""

    @abstractmethod
    async def get_analyst_ratings(self, symbol: str) -> dict[str, Any]:
        """Get analyst ratings and price targets."""

class ISentimentProvider(ABC):
    """Interface for sentiment data providers."""

    @abstractmethod
    async def get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment score for symbol."""

    @abstractmethod
    async def get_social_sentiment(self, symbol: str) -> float:
        """Get social media sentiment score."""

    @abstractmethod
    async def get_market_sentiment(self) -> dict[str, float]:
        """Get overall market sentiment indicators."""

class IBrokerageAdapter(ABC):
    """Interface for brokerage/execution adapters."""

    @abstractmethod
    async def submit_order(self, symbol: str, side: OrderSide, order_type: OrderType, quantity: int, price: float | None=None) -> Order:
        """Submit a trading order."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get current order status."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get current positions."""

    @abstractmethod
    async def get_account_info(self) -> dict[str, Any]:
        """Get account information."""

    @abstractmethod
    async def get_buying_power(self) -> float:
        """Get current buying power."""

class IOrderManager(ABC):
    """Interface for order management."""

    @abstractmethod
    async def process_signal(self, signal: TradingSignal) -> Order | None:
        """Process trading signal and create order."""

    @abstractmethod
    async def manage_order(self, order: Order) -> Order:
        """Manage order lifecycle."""

    @abstractmethod
    async def get_active_orders(self) -> list[Order]:
        """Get all active orders."""

    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """Cancel all active orders."""

class IPositionManager(ABC):
    """Interface for position management."""

    @abstractmethod
    async def update_positions(self) -> list[Position]:
        """Update and return current positions."""

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """Get position for specific symbol."""

    @abstractmethod
    async def calculate_pnl(self) -> dict[str, float]:
        """Calculate profit and loss metrics."""

    @abstractmethod
    async def close_position(self, symbol: str) -> Order | None:
        """Close position for symbol."""

    @abstractmethod
    async def close_all_positions(self) -> list[Order]:
        """Close all positions."""

class IIndicatorCalculator(ABC):
    """Interface for technical indicator calculations."""

    @abstractmethod
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""

    @abstractmethod
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""

    @abstractmethod
    def calculate_rsi(self, data: pd.Series, period: int=14) -> pd.Series:
        """Calculate Relative Strength Index."""

    @abstractmethod
    def calculate_bollinger_bands(self, data: pd.Series, period: int=20, std: float=2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""

    @abstractmethod
    def calculate_macd(self, data: pd.Series, fast: int=12, slow: int=26, signal: int=9) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD."""

class ITradingStrategy(ABC):
    """Interface for trading strategies."""

    @abstractmethod
    async def generate_signal(self, symbol: str, market_data: pd.DataFrame) -> TradingSignal | None:
        """Generate trading signal based on market data."""

    @abstractmethod
    async def update_parameters(self, parameters: dict[str, Any]) -> None:
        """Update strategy parameters."""

    @abstractmethod
    def get_required_data_period(self) -> int:
        """Get required historical data period for strategy."""

    @abstractmethod
    def get_strategy_info(self) -> dict[str, Any]:
        """Get strategy information and current parameters."""

class ISignalGenerator(ABC):
    """Interface for signal generation systems."""

    @abstractmethod
    async def generate_signals(self, symbols: list[str]) -> list[TradingSignal]:
        """Generate signals for multiple symbols."""

    @abstractmethod
    async def add_strategy(self, strategy: ITradingStrategy, weight: float=1.0) -> None:
        """Add trading strategy with weight."""

    @abstractmethod
    async def remove_strategy(self, strategy_name: str) -> bool:
        """Remove trading strategy."""

    @abstractmethod
    def get_active_strategies(self) -> list[str]:
        """Get list of active strategy names."""

class IRiskManager(ABC):
    """Interface for risk management."""

    @abstractmethod
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate if signal meets risk criteria."""

    @abstractmethod
    async def calculate_position_size(self, signal: TradingSignal, account_value: float, current_positions: list[Position]) -> int:
        """Calculate appropriate position size."""

    @abstractmethod
    async def check_risk_limits(self, positions: list[Position], orders: list[Order]) -> dict[str, bool]:
        """Check if current state violates risk limits."""

    @abstractmethod
    async def get_risk_metrics(self) -> dict[str, float]:
        """Get current risk metrics."""

class IPortfolioManager(ABC):
    """Interface for portfolio management."""

    @abstractmethod
    async def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""

    @abstractmethod
    async def calculate_portfolio_metrics(self) -> dict[str, float]:
        """Calculate portfolio performance metrics."""

    @abstractmethod
    async def rebalance_portfolio(self) -> list[TradingSignal]:
        """Generate rebalancing signals."""

    @abstractmethod
    async def get_allocation(self) -> dict[str, float]:
        """Get current portfolio allocation."""

class IMLModel(ABC):
    """Interface for machine learning models."""

    @abstractmethod
    async def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train the model."""

    @abstractmethod
    async def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions."""

    @abstractmethod
    async def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to file."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model from file."""

class IFeatureEngineer(ABC):
    """Interface for feature engineering."""

    @abstractmethod
    async def engineer_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from market data."""

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""

    @abstractmethod
    async def update_features(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Update features with new data."""

class IMetricsCollector(ABC):
    """Interface for metrics collection."""

    @abstractmethod
    async def record_trade(self, order: Order) -> None:
        """Record trade for metrics."""

    @abstractmethod
    async def record_signal(self, signal: TradingSignal) -> None:
        """Record signal for metrics."""

    @abstractmethod
    async def record_performance(self, metrics: dict[str, float]) -> None:
        """Record performance metrics."""

    @abstractmethod
    async def get_metrics(self, start: datetime, end: datetime) -> dict[str, Any]:
        """Get metrics for time period."""

class IAlertManager(ABC):
    """Interface for alert management."""

    @abstractmethod
    async def send_alert(self, level: str, message: str, details: dict[str, Any] | None=None) -> None:
        """Send alert notification."""

    @abstractmethod
    async def configure_alerts(self, config: dict[str, Any]) -> None:
        """Configure alert settings."""

class IHealthMonitor(ABC):
    """Interface for health monitoring."""

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Perform health check."""

    @abstractmethod
    async def get_system_metrics(self) -> dict[str, float]:
        """Get system performance metrics."""

    @abstractmethod
    async def register_component(self, name: str, check_func: callable, interval: int=60) -> None:
        """Register component for health monitoring."""

class IConfigManager(ABC):
    """Interface for configuration management."""

    @abstractmethod
    def get(self, key: str, default: Any=None) -> Any:
        """Get configuration value."""

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""

    @abstractmethod
    def reload(self) -> None:
        """Reload configuration."""

    @abstractmethod
    def validate(self) -> list[str]:
        """Validate configuration and return errors."""

class IStateManager(ABC):
    """Interface for state management."""

    @abstractmethod
    async def save_state(self, state: dict[str, Any]) -> None:
        """Save system state."""

    @abstractmethod
    async def load_state(self) -> dict[str, Any]:
        """Load system state."""

    @abstractmethod
    async def clear_state(self) -> None:
        """Clear saved state."""

class ITradingEngine(ABC):
    """Main trading engine interface."""

    @abstractmethod
    async def start(self) -> None:
        """Start the trading engine."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the trading engine."""

    @abstractmethod
    async def process_signals(self) -> None:
        """Process trading signals."""

    @abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """Get engine status."""

    @abstractmethod
    def is_running(self) -> bool:
        """Check if engine is running."""

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

class SimpleDependencyContainer(IDependencyContainer):
    """Simple implementation of dependency injection container."""

    def __init__(self):
        self._registrations: dict[type, type] = {}
        self._instances: dict[type, Any] = {}
        self._singletons: dict[type, Any] = {}
        self._singleton_flags: dict[type, bool] = {}

    def register(self, interface: type, implementation: type, singleton: bool=False) -> None:
        """Register implementation for interface."""
        self._registrations[interface] = implementation
        self._singleton_flags[interface] = singleton

    def register_instance(self, interface: type, instance: Any) -> None:
        """Register instance for interface."""
        self._instances[interface] = instance

    def resolve(self, interface: type) -> Any:
        """Resolve implementation for interface."""
        if interface in self._instances:
            return self._instances[interface]
        if interface in self._singletons:
            return self._singletons[interface]
        if interface in self._registrations:
            implementation = self._registrations[interface]
            instance = implementation()
            if self._singleton_flags.get(interface, False):
                self._singletons[interface] = instance
            return instance
        raise ValueError(f'No registration found for interface: {interface}')

    def configure(self, config: dict[str, Any]) -> None:
        """Configure container with configuration."""
_container: IDependencyContainer | None = None

def get_container() -> IDependencyContainer:
    """Get or create global dependency container."""
    global _container
    if _container is None:
        _container = SimpleDependencyContainer()
    return _container

def configure_dependencies(config: dict[str, Any]) -> None:
    """Configure dependency injection."""
    container = get_container()
    container.configure(config)

def register_dependencies() -> None:
    """Register default dependencies."""
    get_container()