"""
Trading constants for institutional-grade trading platform.

Contains configuration constants, market parameters, and system limits
used throughout the trading platform.
"""

from datetime import time
from typing import Dict, Any


# Trading session times (US market hours in UTC)
MARKET_HOURS = {
    "PRE_MARKET_START": time(9, 0),    # 9:00 AM UTC (4:00 AM EST)
    "MARKET_OPEN": time(14, 30),       # 2:30 PM UTC (9:30 AM EST)
    "MARKET_CLOSE": time(21, 0),       # 9:00 PM UTC (4:00 PM EST)
    "AFTER_HOURS_END": time(1, 0),     # 1:00 AM UTC (8:00 PM EST)
}

# Risk management parameters
RISK_PARAMETERS = {
    "MAX_PORTFOLIO_RISK": 0.02,        # 2% max portfolio risk per trade
    "MAX_CORRELATION_EXPOSURE": 0.20,   # 20% max exposure to correlated assets
    "MAX_SECTOR_CONCENTRATION": 0.15,   # 15% max exposure to single sector
    "MIN_LIQUIDITY_THRESHOLD": 1000000, # $1M minimum daily volume
    "MAX_POSITION_SIZE": 0.10,          # 10% max position size
    "STOP_LOSS_MULTIPLIER": 2.0,        # 2x ATR for stop loss
    "TAKE_PROFIT_MULTIPLIER": 3.0,      # 3x ATR for take profit
}

# Kelly Criterion parameters
KELLY_PARAMETERS = {
    "MIN_SAMPLE_SIZE": 30,              # Minimum trades for Kelly calculation
    "MAX_KELLY_FRACTION": 0.25,         # Maximum Kelly fraction (25%)
    "CONFIDENCE_LEVEL": 0.95,           # Statistical confidence level
    "LOOKBACK_PERIODS": 252,            # Trading days for analysis
    "REBALANCE_FREQUENCY": 21,          # Rebalance every 21 days
}

# Execution parameters
EXECUTION_PARAMETERS = {
    "MAX_SLIPPAGE_BPS": 20,             # 20 basis points max slippage
    "PARTICIPATION_RATE": 0.10,         # 10% of volume participation
    "MIN_ORDER_SIZE": 100,              # Minimum order size (shares)
    "MAX_ORDER_SIZE": 10000,            # Maximum order size (shares)
    "ORDER_TIMEOUT_SECONDS": 300,       # 5 minute order timeout
    "RETRY_ATTEMPTS": 3,                # Number of retry attempts
    "CANCEL_THRESHOLD_SECONDS": 60,     # Cancel orders after 60 seconds
}

# Data and monitoring parameters
DATA_PARAMETERS = {
    "MAX_DATA_AGE_MINUTES": 5,          # Maximum data age before refresh
    "HEALTH_CHECK_INTERVAL": 60,        # Health check every 60 seconds
    "LOG_ROTATION_DAYS": 30,            # Rotate logs every 30 days
    "METRICS_RETENTION_DAYS": 365,      # Keep metrics for 1 year
    "BACKUP_FREQUENCY_HOURS": 6,        # Backup every 6 hours
}

# Database configuration
DATABASE_PARAMETERS = {
    "CONNECTION_POOL_SIZE": 20,         # SQLAlchemy connection pool size
    "MAX_OVERFLOW": 10,                 # Additional connections allowed
    "POOL_TIMEOUT_SECONDS": 30,         # Connection timeout
    "POOL_RECYCLE_SECONDS": 3600,       # Recycle connections hourly
    "QUERY_TIMEOUT_SECONDS": 30,        # Query timeout
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "MIN_SHARPE_RATIO": 1.0,            # Minimum acceptable Sharpe ratio
    "MAX_DRAWDOWN": 0.20,               # Maximum acceptable drawdown
    "MIN_WIN_RATE": 0.45,               # Minimum win rate threshold
    "MIN_PROFIT_FACTOR": 1.2,           # Minimum profit factor
    "MAX_VAR_95": 0.05,                 # Maximum 95% VaR
}

# System limits
SYSTEM_LIMITS = {
    "MAX_CONCURRENT_ORDERS": 100,       # Maximum concurrent orders
    "MAX_DAILY_TRADES": 1000,           # Maximum trades per day
    "MAX_SYMBOLS_TRACKED": 500,         # Maximum symbols to track
    "MAX_MEMORY_USAGE_GB": 8,           # Maximum memory usage
    "MAX_CPU_USAGE_PERCENT": 80,        # Maximum CPU usage
}

# Consolidate all constants
TRADING_CONSTANTS: Dict[str, Any] = {
    "MARKET_HOURS": MARKET_HOURS,
    "RISK_PARAMETERS": RISK_PARAMETERS,
    "KELLY_PARAMETERS": KELLY_PARAMETERS,
    "EXECUTION_PARAMETERS": EXECUTION_PARAMETERS,
    "DATA_PARAMETERS": DATA_PARAMETERS,
    "DATABASE_PARAMETERS": DATABASE_PARAMETERS,
    "PERFORMANCE_THRESHOLDS": PERFORMANCE_THRESHOLDS,
    "SYSTEM_LIMITS": SYSTEM_LIMITS,
}