# AI Trading Bot
![CI](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/python-app.yml/badge.svg)
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

> **Note**: The `.env` file in this repository contains sample secrets only for
> testing. Real credentials should never be committed to git. In production use
> a secret manager or environment variables provided by your deployment
> platform.

Key environment variables include:

- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` – trading API credentials
- `BOT_MODE` – running mode (`balanced`, `production`, etc.)
- `SLACK_WEBHOOK` – optional webhook URL for alert notifications

### Logging and Alerting

Logs are written to standard output by default so they can be captured by the
systemd journal or Docker logs. If the optional `BOT_LOG_FILE` environment
variable is set a rotating log file handler will also be configured.
Set `SLACK_WEBHOOK` in your environment to enable Slack alerts for critical
errors. Configure logging once at startup:

```python
from logger import setup_logging
import logging

setup_logging(log_file=os.getenv("BOT_LOG_FILE"))
logger = logging.getLogger(__name__)
logger.info("Bot starting up")
```

### Running the Bot

Start the trading bot with:

```bash
python bot.py
```

To expose the webhook server locally run:

```bash
python server.py
```


### Profiling

For performance investigations consider running the bot under `python -m cProfile`
or with `pyinstrument` to identify bottlenecks.

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

Logs are written to stdout and captured by the systemd journal.

## Daily Retraining

The bot can retrain the meta learner each day. To disable this behavior,
set the environment variable `DISABLE_DAILY_RETRAIN=1` before starting the bot.
