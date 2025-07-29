"""Enums and constants for institutional trading system."""

from enum import Enum, auto
from typing import Final


class TradingSide(Enum):
    """Trading position side."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderType(Enum):
    """Order execution types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class AssetClass(Enum):
    """Asset classification for risk management."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    ALTERNATIVE = "alternative"


class MarketRegime(Enum):
    """Market regime classification for strategy adaptation."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class RiskLevel(Enum):
    """Risk classification levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class StrategyType(Enum):
    """Strategy categorization."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    ARBITRAGE = "arbitrage"
    MACHINE_LEARNING = "ml"
    ENSEMBLE = "ensemble"


class TimeFrame(Enum):
    """Trading timeframes."""
    TICK = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class ExecutionStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Trading constants
MAX_POSITION_SIZE: Final[float] = 0.10  # 10% max position size
MAX_PORTFOLIO_LEVERAGE: Final[float] = 2.0  # 2x max leverage
DEFAULT_STOP_LOSS: Final[float] = 0.02  # 2% stop loss
DEFAULT_TAKE_PROFIT: Final[float] = 0.06  # 6% take profit
MIN_LIQUIDITY_THRESHOLD: Final[float] = 100000.0  # $100k min liquidity

# Risk management constants
MAX_DAILY_DRAWDOWN: Final[float] = 0.05  # 5% max daily drawdown
MAX_VAR_95: Final[float] = 0.02  # 2% max 95% VaR
CORRELATION_LIMIT: Final[float] = 0.7  # Max 70% correlation between positions