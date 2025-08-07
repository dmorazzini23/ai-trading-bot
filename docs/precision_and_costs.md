# Precision and Costs Documentation

## Overview

This document describes the profit-critical precision and cost management features implemented to eliminate silent P&L drag and ensure accurate trading cost estimation.

## Money Class and Decimal Precision

### Purpose
The `Money` class provides exact decimal arithmetic for all financial calculations, eliminating floating-point precision errors that can accumulate and cause silent P&L drag.

### Key Features
- Uses Python's `Decimal` type with 28-digit precision
- Banker's rounding (ROUND_HALF_EVEN) for consistent quantization
- Automatic tick size quantization for order execution
- Operator overloading for natural arithmetic operations

### Usage Examples

```python
from ai_trading.math.money import Money, to_decimal, round_to_tick
from decimal import Decimal

# Create Money objects
price = Money('123.567')
quantity = Money('100.0')

# Quantize to tick size
tick_size = Decimal('0.01')
rounded_price = price.quantize(tick_size)
print(rounded_price)  # 123.57

# Arithmetic operations
total_value = price * quantity  # Returns Decimal
commission = Money('1.25')
net_value = total_value - commission  # Returns Money

# Utility functions
market_price = round_to_tick('45.678', Decimal('0.01'))  # Money('45.68')
```

### Determinism Test
The Money class passes the determinism test as required:
```python
assert Money('1.005').quantize(Decimal('0.01')).__str__() in ('1.00','1.01')
```

## Symbol Specifications

### Purpose
Centralized mapping of symbol trading specifications ensures consistent tick and lot sizing across all trading operations.

### Key Features
- Tick size mapping for price quantization
- Lot size mapping for quantity rounding
- Default specifications for major assets
- Runtime specification updates

### Usage Examples

```python
from ai_trading.market.symbol_specs import get_symbol_spec, get_tick_size, get_lot_size

# Get full specification
spec = get_symbol_spec('AAPL')
print(f"AAPL tick: {spec.tick}, lot: {spec.lot}")

# Get individual components
tick = get_tick_size('SPY')    # Decimal('0.01')
lot = get_lot_size('QQQ')      # 1

# Add custom specification
from ai_trading.market.symbol_specs import add_symbol_spec
add_symbol_spec('CUSTOM', Decimal('0.001'), 100)
```

## Enhanced Cost Model

### Purpose
Comprehensive cost modeling including execution costs, short borrow fees, and overnight carry to ensure realistic P&L estimation.

### Cost Components

1. **Execution Costs**
   - Half spread (bid-ask spread impact)
   - Slippage based on volume ratio
   - Commission fees with minimum thresholds

2. **Short Selling Costs**  
   - Borrow fee rates (basis points per day)
   - Hard-to-borrow (HTB) locate requirements
   - Availability checking before short trades

3. **Overnight Costs**
   - Overnight carry charges
   - Position holding cost calculation
   - Multi-day position cost projection

### Usage Examples

```python
from ai_trading.execution.costs import get_symbol_costs, get_cost_model

# Get cost model for symbol
costs = get_symbol_costs('AAPL')
print(f"Half spread: {costs.half_spread_bps} bps")
print(f"Borrow fee: {costs.borrow_fee_bps} bps/day")

# Calculate holding costs
holding_cost = costs.total_holding_cost_bps(days_held=3.0, is_short=True)
print(f"3-day short holding cost: {holding_cost} bps")

# Check short availability
model = get_cost_model()
available, reason = model.check_short_availability('AAPL')
print(f"Short available: {available} ({reason})")

# Position size adjustment for costs
adjusted_size, cost_info = model.adjust_for_holding_costs(
    symbol='AAPL',
    target_size=1000.0,
    expected_holding_days=2.0,
    max_holding_cost_bps=10.0,
    is_short=True
)
print(f"Adjusted size: {adjusted_size} (from 1000)")
```

## Corporate Actions

### Purpose
Unified corporate action adjustment pipeline ensures consistent price and volume adjustments across features, labels, and execution.

### Key Features
- Centralized action registry with persistence
- Automatic price and volume adjustment factors
- Support for splits, dividends, spin-offs, mergers
- Historical data adjustment with reference dates

### Usage Examples

```python
from ai_trading.data.corp_actions import get_corp_action_registry, adjust_bars
import pandas as pd

# Get registry and add action
registry = get_corp_action_registry()
registry.add_action(
    symbol='AAPL',
    ex_date='2020-08-31', 
    action_type='split',
    ratio=4.0,
    description='4-for-1 stock split'
)

# Adjust historical bars
bars_df = pd.DataFrame({
    'open': [100, 105, 102],
    'close': [104, 101, 108],
    'volume': [1000, 1200, 900]
}, index=pd.date_range('2020-08-29', periods=3))

adjusted_bars = adjust_bars(bars_df, 'AAPL')
print("Adjusted for corporate actions:")
print(adjusted_bars)
```

## Integration Points

### Execution Engine Integration
Replace float arithmetic in order sizing and P&L attribution:

```python
from ai_trading.math.money import Money
from ai_trading.market.symbol_specs import get_tick_size

# Order sizing with exact math
target_value = Money('10000.00')  # $10k target
share_price = Money('123.45')
tick_size = get_tick_size('AAPL')

# Calculate shares and round to lot
raw_shares = target_value / share_price
rounded_price = share_price.quantize(tick_size)
```

### Backtesting Integration
Include all cost components in backtest simulation:

```python
from ai_trading.execution.costs import get_symbol_costs

costs = get_symbol_costs('AAPL')

# Execution cost
execution_cost_bps = costs.total_execution_cost_bps(volume_ratio=1.2)

# Holding cost for 2-day position
holding_cost_bps = costs.total_holding_cost_bps(days_held=2.0, is_short=False)

# Total cost impact
total_cost_bps = execution_cost_bps + holding_cost_bps
```

## Configuration

### Cost Model Configuration
Cost parameters can be updated dynamically:

```python
from ai_trading.execution.costs import get_cost_model

model = get_cost_model()

# Update based on realized costs
model.update_costs(
    symbol='AAPL',
    realized_cost_bps=15.2,
    volume_ratio=1.1,
    learning_rate=0.1
)
```

### Symbol Specifications Updates
```python
from ai_trading.market.symbol_specs import update_specs_from_config

# Bulk update from configuration
specs_config = {
    'AAPL': {'tick': '0.01', 'lot': 1},
    'BTC-USD': {'tick': '0.01', 'lot': 1},
    'ES': {'tick': '0.25', 'lot': 1, 'multiplier': 50}
}

update_specs_from_config(specs_config)
```

## Best Practices

1. **Always use Money for financial calculations**
   - Never use float for prices, quantities, or P&L
   - Quantize to appropriate tick size before order submission

2. **Include all cost components**
   - Factor in execution costs, borrow fees, and overnight carry
   - Use realistic slippage models based on volume impact

3. **Maintain corporate action consistency**
   - Apply same adjustments to features, labels, and execution
   - Keep adjustment factors synchronized across systems

4. **Monitor cost model accuracy**
   - Track realized vs expected costs
   - Update model parameters based on execution analysis

5. **Validate precision in testing**
   - Test arithmetic operations for exact results
   - Verify quantization produces expected values
   - Check cost calculations against manual computation