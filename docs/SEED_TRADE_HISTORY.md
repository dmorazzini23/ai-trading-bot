# Seeding Trade History

Meta-learning components rely on historical trade records, but there are two
different file paths involved:

- the seed helper consumes a JSON file, defaulting to
  `trade_history.seed.json`
- the canonical runtime history path is controlled by
  `AI_TRADING_TRADE_HISTORY_PATH`

Deployment can bootstrap the database from the JSON seed file using:

```bash
python -m ai_trading.tools.seed_trade_history  # run manually to verify
```

Before a new deployment, ensure the seed file exists so each symbol has
at least one entry. This avoids warnings and lets the bot collect
statistics from the first trade cycle.

1. Determine the symbols you plan to trade (for example from
   `symbol_override.json`).
2. Create or edit `trade_history.seed.json` or pass an alternate JSON path to
   `python -m ai_trading.tools.seed_trade_history <path>` and add a stub
   record for every symbol:

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
