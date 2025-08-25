# AI Trading Bot

This repository contains a simple trading bot together with a backtesting
harness for optimizing its tunable hyperparameters.

For details on how entry and exit signals are generated, see
[Entry and Exit Signal Methodology](ENTRY_EXIT_SIGNALS.md).

## Running the Backtester

```
python backtester.py --symbols SPY,AAPL --start 2024-01-01 --end 2024-12-31
```

This command runs a grid search over a default parameter grid and writes the best
combination to `best_hyperparams.json`.

### Customizing the Parameter Grid

Edit `backtester.py` and modify the `param_grid` dictionary in `main()` to search
different ranges for each hyperparameter.

## Using Optimized Hyperparameters

When starting the live bot (`python bot_engine.py`), the bot will automatically load
`best_hyperparams.json` if it exists. Otherwise it falls back to the default
values in `hyperparams.json`.

## Development

Install dependencies:

```bash
python -m pip install -U pip
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests

```bash
pytest
```


## Systemd Service

A sample service file `ai-trading.service` is provided. It calls `python -m ai_trading` inside the project virtual environment.

To use it:

```bash
sudo cp packaging/systemd/ai-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading.service
sudo systemctl restart ai-trading.service
sudo systemctl status ai-trading.service
journalctl -u ai-trading.service -n 200 --no-pager
curl -s http://127.0.0.1:9001/healthz
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9001/metrics
```

Ensure `RUN_HEALTHCHECK=1` is set in the environment to expose these endpoints.

The service writes to `logs/scheduler.log` (or `$BOT_LOG_FILE`). View logs with
`tail -F logs/scheduler.log` or via the systemd journal.
