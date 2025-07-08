# 🚀 AI Trading Bot

![CI](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/dmorazzini23/ai-trading-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/dmorazzini23/ai-trading-bot)
[![deploy](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/deploy.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/deploy.yml)

This repository implements a robust **AI-driven trading bot** with:

* a modular backtesting harness for optimizing hyperparameters,
* multi-timeframe technical indicators,
* portfolio scaling via volatility & Kelly constraints,
* and live trading orchestration against Alpaca Markets.

The project targets **Python 3.12.3**.

---

## ⚙️ Installation

Create a virtual environment and install the dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For development & testing, install additional tools:

```bash
pip install -r requirements-dev.txt
```

---

## 📈 Running the Backtester

```bash
python backtest.py --symbols SPY,AAPL --start 2024-01-01 --end 2024-12-31
```

This runs a grid search over a default hyperparameter grid and writes the best configuration to `best_hyperparams.json`.

### 🔧 Customizing the Grid

Edit `backtest.py` to change the `param_grid` dictionary in `main()` for alternative parameter ranges.

---

## 🚀 Using Optimized Hyperparameters

When running live (`python bot_engine.py`), the bot automatically loads `best_hyperparams.json` if available, falling back to `hyperparams.json` otherwise.

---

## 🔑 Configuration

Copy the example environment:

```bash
cp .env.example .env
```

Provide your Alpaca API credentials:

```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
FLASK_PORT=9000
```

Additional key variables include:

* `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`: trading credentials
* `BOT_MODE`: e.g. `balanced`, `production`
* `SLACK_WEBHOOK`: optional for critical alerts
* `BOT_LOG_FILE`: path for rotating logs
* `SCHEDULER_SLEEP_SECONDS`: minimum loop delay (default 30)

> **Note:** `.env` contains only dummy secrets for testing. Never commit real credentials. Use external secrets management in production.

---

## 📝 Logging & Alerting

By default logs write to `logs/scheduler.log` or `$BOT_LOG_FILE` if set, plus stdout for systemd or Docker. Tail them with:

```bash
tail -F logs/scheduler.log
```

Enable Slack alerts by setting `SLACK_WEBHOOK`.

Example logging init:

```python
from logger import setup_logging
import logging

setup_logging(log_file=os.getenv("BOT_LOG_FILE"))
logger = logging.getLogger(__name__)
logger.info("Bot starting up")
```

---

## 🤖 Running the Bot

```bash
python bot_engine.py
```


---

## ⚡ Profiling

For performance investigation:

```bash
python -m cProfile bot_engine.py
# or
pyinstrument python bot_engine.py
```

---

## 🧪 Development & Testing

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests with coverage:

```bash
pytest --cov
```

Expect minimum **90% coverage**.
To drill into indicator prep logic:

```bash
pytest --cov=bot_engine --cov-report=html tests/test_bot_engine*.py
xdg-open htmlcov/index.html
```

---

## 🔥 Systemd Service

Run as a persistent service using the provided unit file:

```bash
sudo cp ai-trading-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading-scheduler.service
```

View logs with:

```bash
tail -F logs/scheduler.log
journalctl -u ai-trading-scheduler.service -f
```

---

## 🔄 Daily Retraining

The bot automatically retrains its meta-learner each day. To disable:

```bash
export DISABLE_DAILY_RETRAIN=1
```

---

## 🤝 AI-Only Maintenance

This project is exclusively maintained by Dom with the help of Codex/GPT-4o.
All automated refactoring, bug fixing, or enhancements are governed by [`AGENTS.md`](./AGENTS.md).
If future AI agents work on this repo, they must strictly follow the directives there to maintain trading safety, logging integrity, and rigorous testing discipline.

---
