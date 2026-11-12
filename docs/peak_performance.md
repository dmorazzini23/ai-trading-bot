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
from ai_trading.execution.idempotency import is_duplicate_order, mark_order_submitted

# Check for duplicates before submission
is_dup, existing_id = is_duplicate_order("AAPL", "buy", 100.0)
if not is_dup:
    order_id = submit_order("AAPL", "buy", 100.0)
    mark_order_submitted("AAPL", "buy", 100.0, order_id)
```

**Key Features:**
- TTL cache with configurable expiration (default 5 minutes)
- Thread-safe operations with RLock
- Bucketed timestamps to handle minor timing differences
- Global cache instance for system-wide deduplication

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
from ai_trading.execution.costs import get_symbol_costs, calculate_execution_cost

# Get cost parameters for symbol
costs = get_symbol_costs("AAPL")
print(f"Half spread: {costs.half_spread_bps}bps, Slippage K: {costs.slip_k}")

# Calculate execution cost for trade
cost_info = calculate_execution_cost("AAPL", position_value=10000, volume_ratio=1.5)
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
from ai_trading.execution.order_policy import create_smart_order

# Create optimized order
order_request = create_smart_order(
    symbol="AAPL",
    side="buy", 
    quantity=100,
    bid=150.25,
    ask=150.30,
    volume_ratio=1.2,
    urgency="medium"
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
# Determinism controls
AI_TRADING_SPEC_OVERRIDE=false    # Allow hash mismatches in development

# Performance tuning
OPENBLAS_NUM_THREADS=1            # Limit BLAS threading
OMP_NUM_THREADS=1                 # Limit OpenMP threading
MKL_NUM_THREADS=1                 # Limit MKL threading
NUMEXPR_NUM_THREADS=1             # Limit NumExpr threading

# Risk controls
RISK_TARGET_VOL=0.10              # 10% volatility target
MAX_TURNOVER_DAILY=1.0            # 100% daily turnover limit
DRAWDOWN_THRESHOLD=0.08           # 8% drawdown threshold

# Cost model
COST_MODEL_VERSION=2.0            # Cost model version
MAX_COST_BPS=20.0                 # Maximum acceptable execution cost
```

### Systemd Service Configuration

```ini
[Unit]
Description=AI Trading Bot
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/opt/ai-trading-bot
ExecStart=/opt/ai-trading-bot/start.sh
Restart=on-failure
RestartSec=10

# Performance environment
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1

[Install]
WantedBy=multi-user.target
```

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
# Lint and format
flake8 ai_trading/
black ai_trading/

# Unit tests
pytest tests/ -v --cov=ai_trading

# Import validation
python -c "import ai_trading; import ai_trading.main"

# Backtest smoke test with cost enforcement
python -m ai_trading.strategies.backtest --smoke --enforce-costs
```

## Best Practices

### Development Workflow

1. **Feature Development**: Use unlocked specification mode
2. **Testing**: Validate with deterministic seeds and hash checking
3. **Staging**: Lock specification and validate compatibility
4. **Production**: Deploy with locked spec and monitoring

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
- `"Feature drift detected"` - Model degradation
- `"High volume detected"` - Order parameter adjustments
- `"Order faded"` - Execution challenges

Use the monitoring dashboard to track:
- Real-time risk metrics
- Cost model performance
- Signal attribution
- System health indicators