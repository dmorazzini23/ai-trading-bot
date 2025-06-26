# AI Trading Bot
![CI](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/dmorazzini23/ai-trading-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/dmorazzini23/ai-trading-bot)
[![deploy](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/deploy.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/deploy.yml)

This repository contains a simple trading bot together with a backtesting
harness for optimizing its tunable hyperparameters.

The project targets **Python 3.12.3**.

## Installation

Create a virtual environment and install dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For development and testing install the additional tools:

```bash
pip install -r requirements-dev.txt
```

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

When starting the live bot (`python bot_engine.py`), the bot will automatically load
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
- `BOT_LOG_FILE` – optional path for a rotating scheduler log file
- `SCHEDULER_SLEEP_SECONDS` – minimum delay between scheduler ticks (default 30)

### Logging and Alerting

Logs are written to `logs/scheduler.log` by default and can be viewed with
`tail -F logs/scheduler.log`. If `BOT_LOG_FILE` is set, that path will be used
instead. Logs are still emitted to stdout for systemd or Docker capture.
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
python bot_engine.py
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
pytest --cov
```

Test coverage should report at least **90%** for a successful run.

To profile coverage for the indicator preparation logic only:

```bash
pytest --cov=bot_engine --cov-report=term --cov-report=html tests/test_bot_engine*.py
```

This outputs a terminal summary and an HTML report at `htmlcov/index.html`.


## Systemd Service

A sample service file `ai-trading-scheduler.service` is provided to run the bot using the `start.sh` helper script. This ensures the virtual environment is activated and all dependencies are installed before the bot starts.

To use it:

```bash
sudo cp ai-trading-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-scheduler.service
```

Logs are written to `logs/scheduler.log` (or `$BOT_LOG_FILE`) and can be tailed
with `tail -F logs/scheduler.log`. They are also output to stdout for the
systemd journal.

## Daily Retraining

The bot can retrain the meta learner each day. To disable this behavior,
set the environment variable `DISABLE_DAILY_RETRAIN=1` before starting the bot.
