# Peak Performance Trading System

This document describes the peak-performance hardening features implemented in the AI trading system, including execution safety, cost modeling, adaptive risk controls, and performance optimizations.

## Overview

The peak-performance system provides:

- **Order Idempotency & Reconciliation**: Prevents duplicate orders and maintains consistency with broker state
- **Time Hygiene & Calendar Alignment**: Ensures proper timing synchronization with exchange schedules
- **Symbol-Aware Cost Modeling**: Tracks and applies per-symbol execution costs
- **Adaptive Risk Controls**: Dynamic position sizing with volatility targeting and correlation clustering
- **Deterministic Training**: Reproducible model training and inference with hash validation
- **Drift Monitoring**: Feature drift detection and signal attribution tracking
- **Performance Optimizations**: Parallel processing and vectorized operations
- **Smart Order Routing**: Intelligent order placement with marketable limit orders

## Components

### 1. Order Idempotency (`ai_trading.execution.idempotency`)

Prevents duplicate order submissions through TTL-based caching:

```python
from ai_trading.execution.idempotency import OrderIdempotencyCache

cache = OrderIdempotencyCache()

# Check for duplicates before submission
key = cache.generate_key("AAPL", "buy", 100.0)
if not cache.is_duplicate(key):
    order_id = submit_order("AAPL", "buy", 100.0)
    cache.mark_submitted(key, order_id)
```

`check_and_mark_submitted(key, order_id)` does the check and the mark atomically.

**Key Features:**
- TTL cache with configurable expiration (default 5 minutes)
- Thread-safe operations with RLock
- Bucketed timestamps to handle minor timing differences

### 2. Position Reconciliation (`ai_trading.execution.reconcile`)

Reconciles local trading state with broker truth:

```python
from ai_trading.execution.reconcile import reconcile_with_broker

# Perform full reconciliation
result = reconcile_with_broker(
    broker_client=alpaca_client,
    local_positions=current_positions,
    local_orders=pending_orders,
    apply_fixes=True
)

print(f"Found {len(result.position_drifts)} position drifts")
print(f"Took {len(result.actions_taken)} corrective actions")
```

**Reconciliation Actions:**
- Position quantity synchronization
- Stale order cleanup
- Order status updates
- Automatic drift correction

### 4. Symbol-Aware Cost Model (`ai_trading.execution.costs`)

Tracks per-symbol execution costs for accurate position sizing:

```python
from ai_trading.execution.costs import get_cost_model, get_symbol_costs

# Get cost parameters for symbol
costs = get_symbol_costs("AAPL")
print(f"Half spread: {costs.half_spread_bps}bps, Slippage K: {costs.slip_k}")

# Calculate execution cost for trade
cost_info = get_cost_model().calculate_position_impact("AAPL", position_value=10000, volume_ratio=1.5)
print(f"Total cost: {cost_info['cost_bps']:.1f}bps (${cost_info['cost_dollars']:.2f})")
```

**Cost Components:**
- Half spread in basis points
- Slippage coefficient (√volume scaling)
- Commission rates
- Minimum commission floors

**Model Updates:**
- Adaptive learning from realized costs
- Nightly snapshots to Parquet files
- Automatic cost model versioning

### 9. Smart Order Routing (`ai_trading.execution.order_policy`)

Intelligent order placement with adaptive strategies:

```python
from ai_trading.execution.order_policy import (
    MarketData,
    OrderUrgency,
    get_smart_router,
)

# Create optimized order
market_data = MarketData(
    symbol="AAPL",
    bid=150.25,
    ask=150.30,
    mid=150.275,
    spread_bps=3.3,
    volume_ratio=1.2,
)
order_request = get_smart_router().create_order_request(
    "AAPL", "buy", 100, market_data, urgency=OrderUrgency.MEDIUM
)

print(f"Order type: {order_request['type']}")
print(f"Limit price: ${order_request['limit_price']:.4f}")
```

**Order Types:**
- **Marketable Limit**: Default for normal conditions
- **IOC (Immediate or Cancel)**: For wide spreads or urgency
- **Market**: Fallback for urgent trades with very wide spreads

**Symbol-Specific Parameters:**
- Spread multiplier (k factor)
- IOC threshold (default 5bps)
- Market fallback threshold (default 20bps)
- Volume adjustments for high-volume periods

## Deployment Configuration

### Environment Variables

```bash
# Performance tuning
OPENBLAS_NUM_THREADS=1            # Limit BLAS threading
OMP_NUM_THREADS=1                 # Limit OpenMP threading
MKL_NUM_THREADS=1                 # Limit MKL threading
NUMEXPR_NUM_THREADS=1             # Limit NumExpr threading

```

Risk, turnover and cost limits are `ConfigSpec` entries in
`ai_trading/config/runtime.py`; list them there rather than relying on ad-hoc
variable names.

### Systemd Service Configuration

The packaged unit is `packaging/systemd/ai-trading.service`; see
`docs/DEPLOYING.md` for deployment.

### Monitoring and Alerts

**Key Metrics to Monitor:**
- Order idempotency cache hit rate
- Position reconciliation drift frequency
- Time skew alerts (>250ms)
- Feature drift PSI scores
- Signal attribution performance
- Execution cost tracking
- Turnover budget utilization

**Alert Thresholds:**
- High feature drift (PSI > 0.2)
- Low hit ratio (<45%)
- High drawdown (>15%)
- Time skew (>250ms)
- Cost model updates (10+ symbols/day)

## Validation and Testing

### Continuous Integration Checks

```bash
# Default post-change validation (see AGENTS.md section 4 for variants)
bash scripts/agent_validate_changed.sh
```

## Best Practices

### Risk Management

1. **Start Conservative**: Begin with lower risk targets and turnover limits
2. **Monitor Correlations**: Watch for cluster concentration
3. **Track Costs**: Ensure cost model updates don't degrade performance
4. **Review Drift**: Investigate medium/high drift features

### Performance Monitoring

1. **Cache Hit Rates**: Monitor idempotency and performance caches
2. **Parallel Efficiency**: Ensure parallel processing improves throughput
3. **Memory Usage**: Watch for memory leaks in long-running processes
4. **I/O Performance**: Use Parquet compression for data persistence

## Troubleshooting

### Common Issues

**High Feature Drift:**
- Check data source changes
- Validate feature engineering pipeline
- Consider retraining or feature selection

**Poor Order Fill Rates:**
- Review order parameters (spread multipliers)
- Check market conditions and volatility
- Consider increasing urgency thresholds

**Performance Degradation:**
- Monitor cache hit rates
- Check parallel processing utilization
- Validate thread limits are set correctly

**Position Reconciliation Errors:**
- Verify broker API connectivity
- Check position tracking logic
- Review order state synchronization

### Support and Monitoring

Monitor log files for key patterns:
- `"Position drift detected"` - Reconciliation issues
- `"High volume detected"` - Order parameter adjustments
- `"Order faded"` - Execution challenges

Use the monitoring dashboard to track:
- Real-time risk metrics
- Cost model performance
- Signal attribution
- System health indicators