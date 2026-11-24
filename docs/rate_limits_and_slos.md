# SLO Monitoring Documentation

## Overview

This document describes the Service Level Objective (SLO) monitoring system. The former central rate limiter (`ai_trading.integrations.rate_limit`) was removed as unused; order pacing lives in the OMS pretrade rate limiter.

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
from ai_trading.monitoring.slo import get_slo_monitor, record_order_reject_rate

# Get SLO monitor
monitor = get_slo_monitor()

# Record metrics
monitor.record_metric('order_latency_ms', 45.2)
monitor.record_metric('position_skew_pct', 1.8)
monitor.record_metric('live_sharpe_ratio', 0.65)

# Convenience function for the order reject rate SLO
record_order_reject_rate(1.5)
```

`record_execution_drift`, `record_realized_slippage` and
`record_order_pacing_cap_hit_rate` are the other module-level helpers.

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