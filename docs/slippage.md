# Slippage Modeling

The execution engine applies a deterministic slippage model defined in `ai_trading/execution/slippage.py`.
Prices are adjusted by a configurable number of basis points (bps) to simulate execution friction.

## Configuration

- `EXECUTION_PARAMETERS["MAX_SLIPPAGE_BPS"]` sets the maximum allowed slippage.
- Override at runtime with the `MAX_SLIPPAGE_BPS` environment variable.
- During tests (`TESTING=1`), the engine raises an error if absolute slippage exceeds this threshold.

## Diagnostics

After each order is filled, the engine logs a `SLIPPAGE_DIAGNOSTIC` entry comparing the expected
submission price with the average fill price. This aids in monitoring execution quality and ensures
that excessive slippage is detected early.
