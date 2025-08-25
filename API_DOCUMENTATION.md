# 📚 API Documentation

## Overview

The AI Trading Bot provides both internal APIs for component interaction and external APIs for monitoring and control. This document covers all available APIs, endpoints, and integration patterns.

## Table of Contents

- [Internal APIs](#internal-apis)
- [Web Interface APIs](#web-interface-apis)
- [Trading APIs](#trading-apis)
- [Data APIs](#data-apis)
- [Monitoring APIs](#monitoring-apis)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

## Internal APIs

### Core Trading Engine API

#### `bot_engine.py` - Main Trading Engine

```python
from bot_engine import pre_trade_health_check, run_all_trades_worker, BotState

def pre_trade_health_check() -> bool:
    """
    Performs comprehensive system health check before trading.
    
    Returns:
        bool: True if system is healthy and ready for trading
        
    Raises:
        SystemError: If critical systems are unavailable
        ValidationError: If configuration is invalid
    """

def run_all_trades_worker(
    symbols: List[str],
    timeframes: List[str] = ['1m', '5m', '15m', '1h', '1d'],
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Execute trading logic for specified symbols and timeframes.
    
    Args:
        symbols: List of trading symbols (e.g., ['SPY', 'AAPL'])
        timeframes: List of timeframes to analyze
        dry_run: If True, simulate trades without execution
        
    Returns:
        Dict containing trade results and performance metrics
        
    Example:
        >>> results = run_all_trades_worker(['SPY'], dry_run=True)
        >>> print(results['trades_executed'])
    """

class BotState:
    """Manages bot operational state and configuration."""
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current portfolio positions."""
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance statistics."""
        
    def update_risk_parameters(self, params: Dict[str, Any]) -> None:
        """Update risk management parameters."""
```

#### `trade_execution.py` - Order Execution

```python
from trade_execution import execute_order_async, validate_order, get_position_info

async def execute_order_async(
    symbol: str,
    quantity: float,
    side: str,
    order_type: str = 'market',
    time_in_force: str = 'day',
    limit_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Execute trading order asynchronously.
    
    Args:
        symbol: Trading symbol (e.g., 'AAPL')
        quantity: Number of shares to trade
        side: 'buy' or 'sell'
        order_type: 'market', 'limit', 'stop'
        time_in_force: 'day', 'gtc', 'ioc', 'fok'
        limit_price: Price for limit orders
        
    Returns:
        Dict containing order status and execution details
        
    Example:
        >>> result = await execute_order_async('AAPL', 10, 'buy')
        >>> print(result['order_id'])
    """

def validate_order(
    symbol: str,
    quantity: float,
    side: str,
    current_portfolio: Dict[str, float]
) -> Tuple[bool, str]:
    """
    Validate order before execution.
    
    Returns:
        Tuple of (is_valid, error_message)
    """

def get_position_info(symbol: str) -> Dict[str, Any]:
    """Get current position information for symbol."""
```

### Data Management API

#### `data_fetcher.py` - Market Data

```python
from ai_trading import data_fetcher

def get_historical_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    provider: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical market data.
    
    Args:
        symbol: Trading symbol
        timeframe: '1m', '5m', '15m', '1h', '1d'
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        provider: 'alpaca', 'finnhub', 'yahoo' (auto-select if None)
        
    Returns:
        DataFrame with OHLCV data
        
    Example:
        >>> data = get_historical_data('SPY', '1h', '2024-01-01', '2024-01-31')
        >>> print(data.head())
    """

async def get_real_time_data(symbols: List[str]) -> Dict[str, Dict]:
    """Get real-time market data for symbols."""

class DataProvider:
    """Abstract base class for data providers."""
    
    def validate_connection(self) -> bool:
        """Check if data provider is accessible."""
        
    def get_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Fetch data from provider."""
```

### Signal Generation API

#### `signals.py` - Trading Signals

```python
from signals import generate_signals, SignalType, SignalStrength

def generate_signals(
    data: pd.DataFrame,
    signal_types: List[str] = None,
    timeframe: str = '1h'
) -> Dict[str, Any]:
    """
    Generate trading signals from market data.
    
    Args:
        data: OHLCV DataFrame
        signal_types: List of signal types to generate
        timeframe: Data timeframe
        
    Returns:
        Dict containing signals and metadata
        
    Example:
        >>> signals = generate_signals(data, ['momentum', 'mean_reversion'])
        >>> print(signals['momentum']['strength'])
    """

class SignalType(Enum):
    """Available signal types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion" 
    MOVING_AVERAGE_CROSSOVER = "ma_crossover"
    REGIME_DETECTION = "regime"

class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2
```

### Risk Management API

#### `risk_engine.py` - Risk Controls

```python
from risk_engine import calculate_position_size, check_risk_limits, RiskMetrics

def calculate_position_size(
    symbol: str,
    signal_strength: float,
    account_equity: float,
    volatility: float,
    max_position_pct: float = 0.05
) -> float:
    """
    Calculate optimal position size using Kelly criterion and volatility.
    
    Args:
        symbol: Trading symbol
        signal_strength: Signal confidence (-1 to 1)
        account_equity: Current account value
        volatility: Symbol volatility (annualized)
        max_position_pct: Maximum position as % of equity
        
    Returns:
        Position size in shares
    """

def check_risk_limits(
    portfolio: Dict[str, float],
    new_position: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Check if new position violates risk limits.
    
    Returns:
        Tuple of (is_within_limits, violation_messages)
    """

class RiskMetrics:
    """Risk calculation utilities."""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk."""
        
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
```

## Web Interface APIs

### Monitoring Dashboard

#### Health Check Endpoint

```http
GET http://127.0.0.1:9001/health
```

Always returns JSON and must never 500.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "components": {
        "database": "healthy",
        "api_connection": "healthy",
        "model_status": "loaded"
    },
    "uptime": "2d 4h 32m",
    "version": "0.1.0"
}
```

#### Performance Metrics

```http
GET /api/metrics
```

**Response:**
```json
{
    "performance": {
        "total_return": 0.15,
        "sharpe_ratio": 1.8,
        "max_drawdown": -0.05,
        "win_rate": 0.65
    },
    "positions": {
        "AAPL": {"shares": 50, "value": 8500.00},
        "SPY": {"shares": 25, "value": 12000.00}
    },
    "recent_trades": [
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 170.50,
            "timestamp": "2024-01-15T09:30:00Z"
        }
    ]
}
```

#### Configuration Management

```http
POST /api/config/update
Content-Type: application/json

{
    "risk_parameters": {
        "max_position_pct": 0.08,
        "max_portfolio_heat": 0.15
    },
    "trading_parameters": {
        "signal_threshold": 0.7,
        "rebalance_frequency": "daily"
    }
}
```

**Response:**
```json
{
    "status": "success",
    "message": "Configuration updated successfully",
    "updated_parameters": ["risk_parameters", "trading_parameters"]
}
```

### Trade Management

#### Get Current Positions

```http
GET /api/positions
```

**Response:**
```json
{
    "positions": [
        {
            "symbol": "AAPL",
            "quantity": 100,
            "market_value": 17000.00,
            "unrealized_pnl": 250.00,
            "cost_basis": 165.00,
            "current_price": 167.50
        }
    ],
    "total_equity": 50000.00,
    "buying_power": 25000.00
}
```

#### Execute Manual Trade

```http
POST /api/trades/execute
Content-Type: application/json

{
    "symbol": "AAPL",
    "quantity": 10,
    "side": "buy",
    "order_type": "market",
    "dry_run": false
}
```

**Response:**
```json
{
    "order_id": "12345678-abcd-1234-5678-123456789abc",
    "status": "filled",
    "symbol": "AAPL",
    "quantity": 10,
    "filled_price": 167.25,
    "commission": 0.00,
    "timestamp": "2024-01-15T10:45:00Z"
}
```

## External Integration APIs

### Webhook Integration

#### Trade Notifications

```http
POST /webhooks/trade-executed
Content-Type: application/json

{
    "event": "trade_executed",
    "data": {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 50,
        "price": 167.50,
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

#### Performance Alerts

```http
POST /webhooks/performance-alert
Content-Type: application/json

{
    "event": "drawdown_alert",
    "data": {
        "current_drawdown": -0.08,
        "threshold": -0.05,
        "portfolio_value": 45000.00,
        "timestamp": "2024-01-15T11:00:00Z"
    }
}
```

## Authentication

### API Key Authentication

All API requests require authentication via API key in the header:

```http
Authorization: Bearer YOUR_API_KEY
```

### Environment Variables

```bash
# Required for API access
API_SECRET_KEY=your_secret_key_here

# Alpaca API credentials
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Error Handling

### Standard Error Response

```json
{
    "error": {
        "code": "INVALID_SYMBOL",
        "message": "Symbol 'INVALID' is not supported",
        "details": {
            "supported_symbols": ["AAPL", "SPY", "QQQ", "..."]
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_SYMBOL` | Unsupported trading symbol | 400 |
| `INSUFFICIENT_FUNDS` | Not enough buying power | 400 |
| `MARKET_CLOSED` | Market is currently closed | 423 |
| `API_RATE_LIMIT` | Rate limit exceeded | 429 |
| `SYSTEM_ERROR` | Internal system error | 500 |
| `API_UNAVAILABLE` | External API unavailable | 503 |

## Rate Limiting

### Default Limits

- **Public endpoints**: 100 requests per minute
- **Trading endpoints**: 50 requests per minute  
- **Webhook endpoints**: 1000 requests per minute

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

### Exceeding Limits

When rate limits are exceeded, a 429 status code is returned:

```json
{
    "error": {
        "code": "API_RATE_LIMIT",
        "message": "Rate limit exceeded. Try again in 60 seconds.",
        "retry_after": 60
    }
}
```

## SDK and Client Libraries

### Python Client

```python
from ai_trading_client import TradingBotClient

client = TradingBotClient(api_key="your_api_key")

# Get current positions
positions = client.get_positions()

# Execute trade
result = client.execute_trade(
    symbol="AAPL",
    quantity=10,
    side="buy"
)

# Get performance metrics
metrics = client.get_performance_metrics()
```

### JavaScript/Node.js Client

```javascript
const TradingBot = require('ai-trading-bot-client');

const client = new TradingBot({
    apiKey: 'your_api_key',
    baseUrl: 'http://localhost:5000'
});

// Get positions
const positions = await client.getPositions();

// Execute trade
const trade = await client.executeTrade({
    symbol: 'AAPL',
    quantity: 10,
    side: 'buy'
});
```

## WebSocket APIs

### Real-time Data Stream

```javascript
const ws = new WebSocket('ws://localhost:5000/ws/data');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'price_update') {
        console.log(`${data.symbol}: $${data.price}`);
    }
};

// Subscribe to symbols
ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['AAPL', 'SPY']
}));
```

### Trade Notifications

```javascript
const ws = new WebSocket('ws://localhost:5000/ws/trades');

ws.onmessage = function(event) {
    const trade = JSON.parse(event.data);
    console.log(`Trade executed: ${trade.symbol} ${trade.side} ${trade.quantity}`);
};
```

## Testing and Development

### Mock API Server

For development and testing, a mock API server is available:

```bash
python -m ai_trading.mock_server --port 5001
```

### Test Data

Sample API responses for testing:

```python
# Test configuration
TEST_CONFIG = {
    "base_url": "http://localhost:5001",
    "api_key": "test_key_123",
    "mock_responses": True
}

# Sample test
def test_get_positions():
    client = TradingBotClient(**TEST_CONFIG)
    positions = client.get_positions()
    assert len(positions) > 0
```

This API documentation provides comprehensive coverage of all available endpoints and integration patterns for the AI Trading Bot system.