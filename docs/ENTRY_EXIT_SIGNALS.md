# Entry and Exit Signal Methodology

This project uses a simple moving-average crossover to derive trading signals.

## Entry (Buy)

A **buy** signal is generated when a short-term simple moving average (SMA)
crosses above a longer-term SMA. The default window lengths are:

- `entry_short_window`: 5 periods
- `entry_long_window`: 20 periods

Both values can be overridden on the runtime context (`ctx`). The functions
operate on a price history DataFrame `df` that must include a `close` column.

## Exit (Sell)

A **sell** signal is produced when the short-term SMA crosses below the
long-term SMA. The default windows mirror the entry settings:

- `exit_short_window`: 5 periods
- `exit_long_window`: 20 periods

The functions return dictionaries with a boolean action flag and the latest
price:

```python
{"buy": True, "price": 123.45}
{"sell": True, "price": 123.45}
```

If no crossover is detected or insufficient data is provided, the functions
return `None`.
