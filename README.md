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
pip install pydantic-settings pydantic>=2.0 python-dateutil>=2.9.2
```

For development & testing, install additional tools:

```bash
pip install -r requirements-dev.txt
```

---

## 📈 Running the Backtester

```bash
python backtester.py --symbols SPY,AAPL --start 2024-01-01 --end 2024-12-31
```

This runs a grid search over a default hyperparameter grid and writes the best configuration to `best_hyperparams.json`.

### 🔧 Customizing the Grid

Edit `backtester.py` to change the `param_grid` dictionary in `main()` for alternative parameter ranges.

---

## 🚀 Using Optimized Hyperparameters

When running live (`python -m ai_trading`), the bot automatically loads `best_hyperparams.json` if available, falling back to `hyperparams.json` otherwise.

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
* `BOT_LOG_FILE`: path for rotating logs
* `SCHEDULER_SLEEP_SECONDS`: delay between scheduler cycles (30–60s recommended)


## Data Sources

Market data is requested from Alpaca first, then Finnhub as a paid secondary source. If both providers fail (e.g. Alpaca subscription errors and Finnhub responds with 403), the bot falls back to `yfinance` for free minute-level data.

---

## 📝 Logging & Alerting

By default logs write to `logs/scheduler.log` or `$BOT_LOG_FILE` if set, plus stdout for systemd or Docker. Tail them with:

```bash
tail -F logs/scheduler.log
```


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
python -m ai_trading
```

The project modules live under `ai_trading/` including `capital_scaling`,
`trade_logic`, and `trade_execution`. Network operations are async; when
calling helpers like `execute_order_async` be sure to run them inside an
event loop:

```python
import asyncio
from trade_execution import execute_order_async

asyncio.run(execute_order_async(...))
```


---

## ⚡ Profiling

For performance investigation:

```bash
python -m cProfile -m ai_trading
# or
pyinstrument python -m ai_trading
```

---

## 🧪 Development & Testing

Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests with coverage:

```bash
pytest -m "not slow" --maxfail=1 --disable-warnings --strict-markers --cov=ai_trading --cov-fail-under=80
```

Slow tests are skipped by default. Run them explicitly with:

```bash
pytest -m slow
```

Expect minimum **80% coverage**.
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
