# Slippage Modeling

The execution engine applies a deterministic slippage model defined in `ai_trading/execution/slippage.py`.
Prices are adjusted by a configurable number of basis points (bps) to simulate execution friction.

## Configuration

- `EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"]` sets the maximum allowed slippage.
- Override at runtime with the `MAX_SLIPPAGE_BPS` environment variable.
- `SLIPPAGE_LIMIT_TOLERANCE_BPS` controls the price buffer when a market order is converted
  to a limit order after the maximum slippage threshold is breached.

## Diagnostics

After each order is filled, the engine logs a `SLIPPAGE_DIAGNOSTIC` entry comparing the expected
submission price with the average fill price. This aids in monitoring execution quality and ensures
that excessive slippage is detected early. When the calculated slippage exceeds the configured
threshold the engine converts market orders to limit orders at `expected_price ± tolerance` or
reduces the order quantity.
