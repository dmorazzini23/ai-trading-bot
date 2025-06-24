# AI Trading Bot

This repository contains a simple trading bot together with a backtesting
harness for optimizing its tunable hyperparameters.

## Running the Backtester

```
python backtest.py --symbols SPY,AAPL --start 2024-01-01 --end 2024-12-31
```

This command runs a grid search over a default parameter grid and writes the best
combination to `best_hyperparams.json`.

### Customizing the Parameter Grid

Edit `backtest.py` and modify the `param_grid` dictionary in `main()` to search
different ranges for each hyperparameter.

## Using Optimized Hyperparameters

When starting the live bot (`python bot.py`), the bot will automatically load
`best_hyperparams.json` if it exists. Otherwise it falls back to the default
values in `hyperparams.json`.

## Development

Install the dependencies listed in `requirements-dev.txt` which in turn
includes `requirements.txt`:

```bash
pip install -r requirements-dev.txt
```

### Running Tests

```bash
pytest
```


## Systemd Service

A sample service file `ai-trading-scheduler.service` is provided to run the bot using the `start.sh` helper script. This ensures the virtual environment is activated and all dependencies are installed before the bot starts.

To use it:

```bash
sudo cp ai-trading-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-scheduler.service
```

The service writes to `logs/scheduler.log` (or `$BOT_LOG_FILE`). View logs with
`tail -F logs/scheduler.log` or via the systemd journal.
