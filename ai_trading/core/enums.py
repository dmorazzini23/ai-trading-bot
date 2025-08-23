"""
Core trading enums for institutional-grade trading platform.

Provides standardized enumerations for order management, risk levels,
and trading operations across the entire platform.
"""
from enum import Enum

class OrderSide(Enum):
    """Order side enumeration for buy/sell operations."""
    BUY = 'buy'
    SELL = 'sell'

    def __str__(self) -> str:
        return self.value

class OrderType(Enum):
    """Order type enumeration for different execution strategies."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'

    def __str__(self) -> str:
        return self.value

class OrderStatus(Enum):
    """Order status enumeration for tracking execution state."""
    PENDING = 'pending'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Check if order status is terminal (no further updates expected)."""
        return self in {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}

class RiskLevel(Enum):
    """Risk level enumeration for position sizing and strategy selection."""
    CONSERVATIVE = 'conservative'
    MODERATE = 'moderate'
    AGGRESSIVE = 'aggressive'

    def __str__(self) -> str:
        return self.value

    @property
    def max_position_size(self) -> float:
        """Maximum position size as fraction of portfolio."""
        mapping = {RiskLevel.CONSERVATIVE: 0.02, RiskLevel.MODERATE: 0.05, RiskLevel.AGGRESSIVE: 0.1}
        return mapping[self]

    @property
    def max_drawdown_threshold(self) -> float:
        """Maximum acceptable drawdown before position reduction."""
        mapping = {RiskLevel.CONSERVATIVE: 0.05, RiskLevel.MODERATE: 0.1, RiskLevel.AGGRESSIVE: 0.15}
        return mapping[self]

class TimeFrame(Enum):
    """Time frame enumeration for market data and analysis."""
    MINUTE_1 = '1m'
    MINUTE_5 = '5m'
    MINUTE_15 = '15m'
    MINUTE_30 = '30m'
    HOUR_1 = '1h'
    HOUR_4 = '4h'
    DAY_1 = '1d'
    WEEK_1 = '1w'

    def __str__(self) -> str:
        return self.value

    @property
    def seconds(self) -> int:
        """Convert timeframe to seconds."""
        mapping = {TimeFrame.MINUTE_1: 60, TimeFrame.MINUTE_5: 300, TimeFrame.MINUTE_15: 900, TimeFrame.MINUTE_30: 1800, TimeFrame.HOUR_1: 3600, TimeFrame.HOUR_4: 14400, TimeFrame.DAY_1: 86400, TimeFrame.WEEK_1: 604800}
        return mapping[self]

class AssetClass(Enum):
    """Asset class enumeration for portfolio diversification."""
    EQUITY = 'equity'
    BOND = 'bond'
    COMMODITY = 'commodity'
    CURRENCY = 'currency'
    CRYPTO = 'crypto'

    def __str__(self) -> str:
        return self.value