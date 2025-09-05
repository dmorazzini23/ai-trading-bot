# Seeding Trade History

Meta-learning components rely on a history file to evaluate signal
performance. Deployment now bootstraps the database from
`trade_history.json` using a helper script invoked during startup:

```bash
python -m ai_trading.tools.seed_trade_history  # run manually to verify
```

Before a new deployment, ensure the history file exists so each symbol has
at least one entry. This avoids warnings and lets the bot collect
statistics from the first trade cycle.

1. Determine the symbols you plan to trade (for example from
   `symbol_override.json`).
2. Create or edit `trade_history.json` (or the path specified by
   `TRADE_LOG_PATH`) and add a stub record for every symbol:

```json
[
  {
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 0,
    "entry_price": 0,
    "exit_price": 0,
    "entry_time": "1970-01-01T00:00:00",
    "exit_time": "1970-01-01T00:00:00",
    "pnl": 0,
    "confidence": 0.0,
    "signal_strength": 0.0
  }
]
```

Stub entries use zero values and a dummy timestamp but reserve space for
real trade data. Replace the example symbol with one entry per ticker in
your universe.

The bot will now log an informational message if the history is empty and
continue with default weights, enabling smooth dry runs and first-time
deployments.

