# Rate Limits and SLOs Documentation

## Overview

This document describes the rate limiting and Service Level Objective (SLO) monitoring systems designed to prevent API throttling and ensure system performance meets defined standards.

## Central Rate Limiter

### Purpose
Orchestrate API calls across the entire trading system to prevent 429 errors and API throttling while providing configurable limits per route with burst capacity.

### Architecture
- **Token Bucket Algorithm** - Allows burst capacity with controlled refill
- **Per-Route Limits** - Different limits for orders, market data, account info
- **Global Rate Limiting** - System-wide capacity management
- **Async-Safe Interface** - Thread-safe for concurrent operations
- **Jittered Refill** - Prevents thundering herd effects

### Rate Limit Configuration

Default route configurations:

| Route | Burst Capacity | Refill Rate | Use Case |
|-------|----------------|-------------|----------|
| orders | 50 | 10/sec | Order submission/modification |
| bars | 200 | 50/sec | Historical data requests |
| quotes | 100 | 20/sec | Real-time quote requests |
| positions | 20 | 5/sec | Position/portfolio queries |
| account | 10 | 2/sec | Account information |

### Usage Examples

#### Async Rate Limiting
```python
from ai_trading.integrations.rate_limit import get_rate_limiter

# Get global rate limiter
limiter = get_rate_limiter()

# Async usage
async def submit_order():
    # Acquire tokens before API call
    acquired = await limiter.acquire('orders', tokens=1, timeout=30.0)
    if not acquired:
        raise Exception("Rate limit exceeded")
    
    # Make API call
    response = await api_client.submit_order(...)
    return response

# Context manager usage
async def get_quotes():
    async with limiter.limit('quotes', tokens=1):
        quotes = await api_client.get_quotes(...)
        return quotes
```

#### Synchronous Rate Limiting
```python
from ai_trading.integrations.rate_limit import rate_limit_sync

# Synchronous usage
def fetch_historical_data():
    acquired = rate_limit_sync('bars', tokens=5, timeout=10.0)
    if not acquired:
        logger.warning("Rate limit exceeded for historical data")
        return None
    
    return api_client.get_historical_bars(...)
```

#### Custom Route Configuration
```python
from ai_trading.integrations.rate_limit import RateLimitConfig

# Define custom rate limits
custom_config = RateLimitConfig(
    capacity=30,        # 30 burst capacity
    refill_rate=5.0,    # 5 tokens per second
    queue_timeout=60.0, # 60 second max wait
    enabled=True
)

limiter.configure_route('custom_api', custom_config)
```

### Rate Limiter Status
```python
# Check rate limiter status
status = limiter.get_status()
print(f"Global available tokens: {status['global']['available_tokens']}")

# Check specific route
orders_status = limiter.get_status('orders')
print(f"Orders available: {orders_status['available_tokens']}")
print(f"Orders capacity: {orders_status['capacity']}")

# Get metrics
print(f"Orders metrics: {orders_status['metrics']}")
```

### Integration with API Clients
```python
class AlpacaClientWithRateLimit:
    def __init__(self):
        self.limiter = get_rate_limiter()
        self.client = AlpacaAPI()
    
    async def submit_order(self, order_data):
        # Rate limit order submissions
        async with self.limiter.limit('orders'):
            return await self.client.submit_order(order_data)
    
    async def get_bars(self, symbols, timeframe):
        # Rate limit market data requests
        tokens_needed = len(symbols)  # One token per symbol
        acquired = await self.limiter.acquire('bars', tokens=tokens_needed)
        if not acquired:
            raise RateLimitExceeded("Market data rate limit exceeded")
        
        return await self.client.get_bars(symbols, timeframe)
```

## SLO Monitoring

### Purpose
Monitor system performance against defined Service Level Objectives and trigger alerts and circuit breakers when thresholds are breached.

### SLO Definitions

Default SLO thresholds:

| Metric | Warning | Critical | Breach | Window | Description |
|--------|---------|----------|--------|--------|-------------|
| Order Latency | 100ms | 500ms | 1000ms | 5min | Order execution time |
| Position Skew | 2% | 5% | 10% | 1min | Position vs target |
| Turnover Ratio | 1.5x | 2.0x | 3.0x | 10min | Trading turnover |
| Live Sharpe | 0.3 | 0.0 | -0.5 | 60min | Risk-adjusted returns |
| Error Rate | 1% | 5% | 10% | 5min | System error percentage |
| Data Staleness | 2min | 5min | 10min | 1min | Market data age |
| P&L Drift | 10bps | 25bps | 50bps | 15min | Attribution drift |

### SLO Status Levels
- **Healthy** - Performance within normal thresholds
- **Warning** - Performance degraded but acceptable
- **Critical** - Performance significantly degraded
- **Breached** - Performance unacceptable, circuit breakers triggered

### Usage Examples

#### Recording Metrics
```python
from ai_trading.monitoring.slo import get_slo_monitor, record_latency

# Get SLO monitor
monitor = get_slo_monitor()

# Record metrics
monitor.record_metric('order_latency_ms', 45.2)
monitor.record_metric('position_skew_pct', 1.8)
monitor.record_metric('live_sharpe_ratio', 0.65)

# Convenience functions
record_latency('order_execution', 67.3)
record_performance_metric('turnover_ratio', 1.2, tags={'strategy': 'momentum'})
```

#### Checking SLO Status
```python
# Get overall health summary
health = monitor.get_health_summary()
print(f"Overall health: {health['overall_health']}")
print(f"Status counts: {health['status_counts']}")

# Get specific SLO status
order_slo = monitor.get_slo_status('order_latency_ms')
print(f"Order latency status: {order_slo['status']}")
print(f"Current value: {order_slo['current_value']}ms")

# Get all SLO statuses
all_slos = monitor.get_slo_status()
for name, status in all_slos.items():
    print(f"{name}: {status['status']} ({status['current_value']})")
```

#### Alert History
```python
# Get recent alerts
alerts = monitor.get_alerts(limit=5)
for alert in alerts:
    print(f"{alert['timestamp']}: {alert['type']} - {alert['metric']}")
    print(f"  {alert['old_status']} -> {alert['new_status']}")
    print(f"  Value: {alert['current_value']}")
```

### Circuit Breakers

#### Built-in Circuit Breakers
```python
from ai_trading.monitoring.slo import setup_default_circuit_breakers

# Setup default circuit breakers
setup_default_circuit_breakers()

# This registers:
# - Trading pause on critical latency/error rate
# - Position size reduction on performance degradation
```

#### Custom Circuit Breakers
```python
def emergency_stop_circuit_breaker(metric_name: str, value: float):
    """Emergency stop on critical SLO breach."""
    logger.critical(f"EMERGENCY STOP: {metric_name} = {value}")
    # Implement emergency stop logic
    trading_engine.emergency_stop()
    notification_service.send_alert(
        level='CRITICAL',
        message=f"Emergency stop triggered by {metric_name}"
    )

def reduce_risk_budget_circuit_breaker(metric_name: str, value: float):
    """Reduce risk budget on SLO degradation."""
    logger.warning(f"Reducing risk budget due to {metric_name} = {value}")
    # Reduce position sizing
    risk_manager.scale_risk_budget(0.5)  # 50% reduction

# Register custom circuit breakers
monitor.register_circuit_breaker('order_latency_ms', emergency_stop_circuit_breaker)
monitor.register_circuit_breaker('live_sharpe_ratio', reduce_risk_budget_circuit_breaker)
```

### Custom SLO Thresholds
```python
from ai_trading.monitoring.slo import SLOThreshold

# Define custom SLO
custom_slo = SLOThreshold(
    name='strategy_alpha',
    warning_threshold=0.1,    # 10bps alpha warning
    critical_threshold=0.0,   # 0bps alpha critical  
    breach_threshold=-0.1,    # -10bps alpha breach
    window_minutes=30,
    min_samples=10,
    description='Strategy alpha generation'
)

monitor.add_slo_threshold(custom_slo)

# Record alpha measurements
monitor.record_metric('strategy_alpha', 0.05)  # 5bps alpha
```

## Integration Patterns

### API Client Integration
```python
class RateLimitedAPIClient:
    def __init__(self, api_client):
        self.client = api_client
        self.limiter = get_rate_limiter()
        self.monitor = get_slo_monitor()
    
    async def api_call_with_monitoring(self, route: str, operation: str, *args, **kwargs):
        # Rate limiting
        start_time = time.time()
        acquired = await self.limiter.acquire(route, timeout=30.0)
        if not acquired:
            self.monitor.record_metric('error_rate_pct', 100.0)
            raise RateLimitExceeded(f"Rate limit exceeded for {route}")
        
        try:
            # Make API call
            result = await getattr(self.client, operation)(*args, **kwargs)
            
            # Record successful latency
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_metric(f'{operation}_latency_ms', latency_ms)
            self.monitor.record_metric('error_rate_pct', 0.0)
            
            return result
            
        except Exception as e:
            # Record error
            self.monitor.record_metric('error_rate_pct', 100.0)
            logger.error(f"API call failed: {operation} - {e}")
            raise
```

### Trading Engine Integration
```python
class TradingEngineWithSLO:
    def __init__(self):
        self.monitor = get_slo_monitor()
        self.engine = TradingEngine()
    
    def execute_trade(self, order):
        start_time = time.time()
        
        try:
            # Execute order
            result = self.engine.execute_order(order)
            
            # Record execution metrics
            latency_ms = (time.time() - start_time) * 1000
            self.monitor.record_metric('order_latency_ms', latency_ms)
            
            # Record position skew
            target_position = self.get_target_position(order.symbol)
            current_position = self.get_current_position(order.symbol)
            skew_pct = abs(current_position - target_position) / abs(target_position) * 100
            self.monitor.record_metric('position_skew_pct', skew_pct)
            
            return result
            
        except Exception as e:
            # Record execution failure
            self.monitor.record_metric('error_rate_pct', 100.0)
            raise
```

## Configuration Files

### Rate Limit Configuration
```json
{
  "global": {
    "capacity": 1000,
    "refill_rate": 100.0
  },
  "routes": {
    "orders": {
      "capacity": 50,
      "refill_rate": 10.0,
      "queue_timeout": 30.0,
      "enabled": true
    },
    "market_data": {
      "capacity": 200,
      "refill_rate": 50.0,
      "queue_timeout": 10.0,
      "enabled": true
    }
  }
}
```

### SLO Configuration
```json
{
  "slos": [
    {
      "name": "order_latency_ms",
      "warning_threshold": 100.0,
      "critical_threshold": 500.0,
      "breach_threshold": 1000.0,
      "window_minutes": 5,
      "min_samples": 5,
      "description": "Order execution latency"
    },
    {
      "name": "custom_metric",
      "warning_threshold": 10.0,
      "critical_threshold": 20.0,
      "breach_threshold": 50.0,
      "window_minutes": 10,
      "min_samples": 3,
      "description": "Custom business metric"
    }
  ]
}
```

## Monitoring and Alerting

### Health Check Endpoint
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health_check():
    monitor = get_slo_monitor()
    limiter = get_rate_limiter()
    
    health_summary = monitor.get_health_summary()
    rate_limit_status = limiter.get_status()
    
    return jsonify({
        'status': health_summary['overall_health'],
        'slo_summary': health_summary,
        'rate_limits': rate_limit_status,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
```

### Metrics Export
```python
def export_metrics_to_prometheus():
    """Export SLO metrics to Prometheus format."""
    monitor = get_slo_monitor()
    
    metrics = []
    for name, status in monitor.get_slo_status().items():
        metrics.append(f'slo_status{{metric="{name}"}} {status["current_value"]}')
        metrics.append(f'slo_threshold_warning{{metric="{name}"}} {status["threshold"]["warning"]}')
    
    return '\n'.join(metrics)
```

## Best Practices

### Rate Limiting
1. **Configure appropriate burst capacity** for API patterns
2. **Use timeouts** to prevent indefinite blocking
3. **Monitor rate limit metrics** to optimize settings
4. **Implement graceful degradation** when limits are hit
5. **Use jittered delays** to prevent thundering herd

### SLO Monitoring
1. **Set realistic thresholds** based on historical performance
2. **Use appropriate window sizes** for different metrics
3. **Implement gradual alerting** (warning → critical → breach)
4. **Test circuit breakers** in non-production environments
5. **Review and adjust thresholds** based on operational experience

### Circuit Breakers
1. **Implement multiple response levels** (warning, critical, emergency)
2. **Test circuit breaker logic** thoroughly
3. **Document circuit breaker procedures** for operations team
4. **Monitor circuit breaker activation frequency**
5. **Provide manual override capabilities** for emergency situations

### Integration
1. **Centralize rate limiting** across all API clients
2. **Standardize SLO metrics** across components
3. **Use consistent naming conventions** for metrics
4. **Implement comprehensive logging** for debugging
5. **Provide operational dashboards** for monitoring