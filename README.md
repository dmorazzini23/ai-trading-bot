# AI Trading Bot
![CI](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/dmorazzini23/ai-trading-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/dmorazzini23/ai-trading-bot)

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

## Configuration

Copy `.env.example` to `.env` and provide your Alpaca API credentials:

```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
FLASK_PORT=9000
```

Only these variables are required for Alpaca access.

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

The service writes logs to `/var/log/ai-trading-scheduler.log`.

## Daily Retraining

The bot can retrain the meta learner each day. To disable this behavior,
set the environment variable `DISABLE_DAILY_RETRAIN=1` before starting the bot.
