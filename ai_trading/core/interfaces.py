"""Clean architecture interfaces for dependency injection and separation of concerns.

Provides abstract interfaces for all major components to enable
proper dependency injection, testing, and modular design.

AI-AGENT-REF: Clean architecture interfaces for production trading platform
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, AsyncIterator
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np


# Core Trading Types
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    symbol: str
    side: OrderSide
    confidence: float
    timestamp: datetime
    price: Optional[float] = None
    quantity: Optional[int] = None
    metadata: Dict[str, Any] = None


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: int
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


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
    price: Optional[float]
    filled_price: Optional[float]
    timestamp: datetime
    metadata: Dict[str, Any] = None


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
    metadata: Dict[str, Any] = None


# Data Provider Interfaces

class IDataProvider(ABC):
    """Interface for market data providers."""
    
    @abstractmethod
    async def get_market_data(self, 
                            symbol: str, 
                            timeframe: str = "1min",
                            start: Optional[datetime] = None,
                            end: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical market data."""
        pass
    
    @abstractmethod
    async def get_real_time_data(self, symbol: str) -> AsyncIterator[MarketData]:
        """Get real-time market data stream."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        pass
    
    @abstractmethod
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status information."""
        pass


class IFundamentalDataProvider(ABC):
    """Interface for fundamental data providers."""
    
    @abstractmethod
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company fundamental information."""
        pass
    
    @abstractmethod
    async def get_financial_statements(self, symbol: str) -> Dict[str, Any]:
        """Get financial statements."""
        pass
    
    @abstractmethod
    async def get_analyst_ratings(self, symbol: str) -> Dict[str, Any]:
        """Get analyst ratings and price targets."""
        pass


class ISentimentProvider(ABC):
    """Interface for sentiment data providers."""
    
    @abstractmethod
    async def get_news_sentiment(self, symbol: str) -> float:
        """Get news sentiment score for symbol."""
        pass
    
    @abstractmethod
    async def get_social_sentiment(self, symbol: str) -> float:
        """Get social media sentiment score."""
        pass
    
    @abstractmethod
    async def get_market_sentiment(self) -> Dict[str, float]:
        """Get overall market sentiment indicators."""
        pass


# Trading Engine Interfaces

class IBrokerageAdapter(ABC):
    """Interface for brokerage/execution adapters."""
    
    @abstractmethod
    async def submit_order(self, 
                         symbol: str,
                         side: OrderSide,
                         order_type: OrderType,
                         quantity: int,
                         price: Optional[float] = None) -> Order:
        """Submit a trading order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get current order status."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_buying_power(self) -> float:
        """Get current buying power."""
        pass


class IOrderManager(ABC):
    """Interface for order management."""
    
    @abstractmethod
    async def process_signal(self, signal: TradingSignal) -> Optional[Order]:
        """Process trading signal and create order."""
        pass
    
    @abstractmethod
    async def manage_order(self, order: Order) -> Order:
        """Manage order lifecycle."""
        pass
    
    @abstractmethod
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        pass
    
    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """Cancel all active orders."""
        pass


class IPositionManager(ABC):
    """Interface for position management."""
    
    @abstractmethod
    async def update_positions(self) -> List[Position]:
        """Update and return current positions."""
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        pass
    
    @abstractmethod
    async def calculate_pnl(self) -> Dict[str, float]:
        """Calculate profit and loss metrics."""
        pass
    
    @abstractmethod
    async def close_position(self, symbol: str) -> Optional[Order]:
        """Close position for symbol."""
        pass
    
    @abstractmethod
    async def close_all_positions(self) -> List[Order]:
        """Close all positions."""
        pass


# Strategy and Signal Generation Interfaces

class IIndicatorCalculator(ABC):
    """Interface for technical indicator calculations."""
    
    @abstractmethod
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        pass
    
    @abstractmethod
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        pass
    
    @abstractmethod
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        pass
    
    @abstractmethod
    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        pass
    
    @abstractmethod
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD."""
        pass


class ITradingStrategy(ABC):
    """Interface for trading strategies."""
    
    @abstractmethod
    async def generate_signal(self, symbol: str, market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Generate trading signal based on market data."""
        pass
    
    @abstractmethod
    async def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        pass
    
    @abstractmethod
    def get_required_data_period(self) -> int:
        """Get required historical data period for strategy."""
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current parameters."""
        pass


class ISignalGenerator(ABC):
    """Interface for signal generation systems."""
    
    @abstractmethod
    async def generate_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate signals for multiple symbols."""
        pass
    
    @abstractmethod
    async def add_strategy(self, strategy: ITradingStrategy, weight: float = 1.0) -> None:
        """Add trading strategy with weight."""
        pass
    
    @abstractmethod
    async def remove_strategy(self, strategy_name: str) -> bool:
        """Remove trading strategy."""
        pass
    
    @abstractmethod
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy names."""
        pass


# Risk Management Interfaces

class IRiskManager(ABC):
    """Interface for risk management."""
    
    @abstractmethod
    async def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate if signal meets risk criteria."""
        pass
    
    @abstractmethod
    async def calculate_position_size(self, 
                                    signal: TradingSignal,
                                    account_value: float,
                                    current_positions: List[Position]) -> int:
        """Calculate appropriate position size."""
        pass
    
    @abstractmethod
    async def check_risk_limits(self, 
                              positions: List[Position],
                              orders: List[Order]) -> Dict[str, bool]:
        """Check if current state violates risk limits."""
        pass
    
    @abstractmethod
    async def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        pass


class IPortfolioManager(ABC):
    """Interface for portfolio management."""
    
    @abstractmethod
    async def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        pass
    
    @abstractmethod
    async def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        pass
    
    @abstractmethod
    async def rebalance_portfolio(self) -> List[TradingSignal]:
        """Generate rebalancing signals."""
        pass
    
    @abstractmethod
    async def get_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation."""
        pass


# Machine Learning Interfaces

class IMLModel(ABC):
    """Interface for machine learning models."""
    
    @abstractmethod
    async def train(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    async def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    async def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to file."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model from file."""
        pass


class IFeatureEngineer(ABC):
    """Interface for feature engineering."""
    
    @abstractmethod
    async def engineer_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from market data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        pass
    
    @abstractmethod
    async def update_features(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Update features with new data."""
        pass


# Monitoring and Logging Interfaces

class IMetricsCollector(ABC):
    """Interface for metrics collection."""
    
    @abstractmethod
    async def record_trade(self, order: Order) -> None:
        """Record trade for metrics."""
        pass
    
    @abstractmethod
    async def record_signal(self, signal: TradingSignal) -> None:
        """Record signal for metrics."""
        pass
    
    @abstractmethod
    async def record_performance(self, metrics: Dict[str, float]) -> None:
        """Record performance metrics."""
        pass
    
    @abstractmethod
    async def get_metrics(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get metrics for time period."""
        pass


class IAlertManager(ABC):
    """Interface for alert management."""
    
    @abstractmethod
    async def send_alert(self, 
                       level: str,
                       message: str,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """Send alert notification."""
        pass
    
    @abstractmethod
    async def configure_alerts(self, config: Dict[str, Any]) -> None:
        """Configure alert settings."""
        pass


class IHealthMonitor(ABC):
    """Interface for health monitoring."""
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Perform health check."""
        pass
    
    @abstractmethod
    async def get_system_metrics(self) -> Dict[str, float]:
        """Get system performance metrics."""
        pass
    
    @abstractmethod
    async def register_component(self, 
                                name: str,
                                check_func: callable,
                                interval: int = 60) -> None:
        """Register component for health monitoring."""
        pass


# Configuration and State Management Interfaces

class IConfigManager(ABC):
    """Interface for configuration management."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    @abstractmethod
    def reload(self) -> None:
        """Reload configuration."""
        pass
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate configuration and return errors."""
        pass


class IStateManager(ABC):
    """Interface for state management."""
    
    @abstractmethod
    async def save_state(self, state: Dict[str, Any]) -> None:
        """Save system state."""
        pass
    
    @abstractmethod
    async def load_state(self) -> Dict[str, Any]:
        """Load system state."""
        pass
    
    @abstractmethod
    async def clear_state(self) -> None:
        """Clear saved state."""
        pass


# Main Trading Engine Interface

class ITradingEngine(ABC):
    """Main trading engine interface."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the trading engine."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the trading engine."""
        pass
    
    @abstractmethod
    async def process_signals(self) -> None:
        """Process trading signals."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if engine is running."""
        pass


# Dependency Injection Container

class IDependencyContainer(ABC):
    """Interface for dependency injection container."""
    
    @abstractmethod
    def register(self, interface: type, implementation: type, singleton: bool = False) -> None:
        """Register implementation for interface."""
        pass
    
    @abstractmethod
    def register_instance(self, interface: type, instance: Any) -> None:
        """Register instance for interface."""
        pass
    
    @abstractmethod
    def resolve(self, interface: type) -> Any:
        """Resolve implementation for interface."""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure container with configuration."""
        pass


class SimpleDependencyContainer(IDependencyContainer):
    """Simple implementation of dependency injection container."""
    
    def __init__(self):
        self._registrations: Dict[type, type] = {}
        self._instances: Dict[type, Any] = {}
        self._singletons: Dict[type, Any] = {}
        self._singleton_flags: Dict[type, bool] = {}
    
    def register(self, interface: type, implementation: type, singleton: bool = False) -> None:
        """Register implementation for interface."""
        self._registrations[interface] = implementation
        self._singleton_flags[interface] = singleton
    
    def register_instance(self, interface: type, instance: Any) -> None:
        """Register instance for interface."""
        self._instances[interface] = instance
    
    def resolve(self, interface: type) -> Any:
        """Resolve implementation for interface."""
        # Check for registered instance first
        if interface in self._instances:
            return self._instances[interface]
        
        # Check for singleton
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Check for registered implementation
        if interface in self._registrations:
            implementation = self._registrations[interface]
            instance = implementation()
            
            # Store as singleton if configured
            if self._singleton_flags.get(interface, False):
                self._singletons[interface] = instance
            
            return instance
        
        raise ValueError(f"No registration found for interface: {interface}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure container with configuration."""
        # This could configure implementations based on config
        pass


# Global container instance
_container: Optional[IDependencyContainer] = None


def get_container() -> IDependencyContainer:
    """Get or create global dependency container."""
    global _container
    if _container is None:
        _container = SimpleDependencyContainer()
    return _container


def configure_dependencies(config: Dict[str, Any]) -> None:
    """Configure dependency injection."""
    container = get_container()
    container.configure(config)


def register_dependencies() -> None:
    """Register default dependencies."""
    container = get_container()
    
    # Register default implementations here when available
    # container.register(IDataProvider, AlpacaDataProvider, singleton=True)
    # container.register(IBrokerageAdapter, AlpacaBrokerageAdapter, singleton=True)
    # etc.
    
    pass