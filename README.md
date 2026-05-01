# AI Trading Bot

[![CI](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/ci.yml)
[![CodeQL](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/codeql.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/codeql.yml)
[![Dependency Audit](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/dependency-audit.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/dependency-audit.yml)
[![SBOM](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/sbom.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/sbom.yml)
[![Scorecard](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/scorecard.yml/badge.svg)](https://github.com/dmorazzini23/ai-trading-bot/actions/workflows/scorecard.yml)
[![codecov](https://codecov.io/gh/dmorazzini23/ai-trading-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/dmorazzini23/ai-trading-bot)
[![Python 3.12.3](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI Trading Bot is a Python 3.12 algorithmic trading runtime for Alpaca Markets.
It combines market-data ingestion, technical indicators, machine-learning
signals, guarded order execution, risk controls, Prometheus metrics, and
operator health endpoints.

This project is for research and operator-supervised paper/live trading. It
does not guarantee profitability. Start in paper mode, validate your own
configuration, and never trade with money you cannot afford to lose.

## Current Runtime Contract

- Runtime: Ubuntu 24.04, Python 3.12.3, `alpaca-py==0.42.1`.
- Timezones: stdlib `zoneinfo` only.
- Main API: Flask on `0.0.0.0:9001` by default.
- Health and metrics: `/health`, `/healthz`, and `/metrics` on `API_PORT`.
- Standalone health app: `RUN_HEALTHCHECK=1 ./venv/bin/python -m ai_trading.app`
  on `HEALTHCHECK_PORT` (default `8081`).
- Configuration access: `ai_trading.config.management`.
- Production service: `packaging/systemd/ai-trading.service`.

## Quickstart

```bash
python3.12 -m venv venv
. venv/bin/activate
./venv/bin/python -m pip install -U pip
./venv/bin/python -m pip install -e .

./venv/bin/python -c "import importlib.metadata as m; assert m.version('alpaca-py') == '0.42.1'"
./venv/bin/python -m ai_trading --dry-run
```

Configure paper-trading credentials before running the scheduler:

```bash
cp .env.example .env
$EDITOR .env
./venv/bin/python -m ai_trading.tools.env_validate
```

Minimum required runtime settings are documented in [DEPLOYING.md](DEPLOYING.md)
and [docs/API_KEY_SETUP.md](docs/API_KEY_SETUP.md). The canonical Alpaca broker
endpoint variable is `ALPACA_TRADING_BASE_URL`; deprecated `ALPACA_API_URL` and
`ALPACA_BASE_URL` are rejected.

Run one paper cycle:

```bash
./venv/bin/python -m ai_trading --once --paper
```

Start the main runtime:

```bash
./venv/bin/python -m ai_trading
curl -s http://127.0.0.1:9001/healthz
curl -s http://127.0.0.1:9001/metrics | head
```

Run the standalone health app when you need an independent liveness surface:

```bash
RUN_HEALTHCHECK=1 ./venv/bin/python -m ai_trading.app &
curl -s http://127.0.0.1:${HEALTHCHECK_PORT:-8081}/healthz
```

## Development Checks

Use the repository virtualenv. Do not run repo-local Python, pytest, pip, or git
commands with `sudo`.

```bash
./venv/bin/ruff check
./venv/bin/python -m py_compile $(git ls-files '*.py')
./venv/bin/pytest -q
./venv/bin/mypy
bash scripts/typecheck_strict.sh
```

During market hours, prefer targeted tests and live runtime validation unless a
broad test run is explicitly approved.

## Common Workflows

| Task | Command |
| --- | --- |
| Validate config | `./venv/bin/python -m ai_trading.tools.env_validate` |
| Alpaca self-check | `./venv/bin/python ai_trading/scripts/self_check.py` |
| Research backtest | `./venv/bin/python -m ai_trading.strategies.backtester --symbols SPY QQQ --data-dir ./data --start 2024-01-01 --end 2024-03-31` |
| Offline replay | `./venv/bin/python -m ai_trading.tools.offline_replay --data-dir ./data --symbols SPY,AAPL --output-json artifacts/offline_replay.json` |
| Retrain meta learner | `./venv/bin/python -m ai_trading.retrain --trade-log data/trades.csv --model-path runtime/meta_model.joblib` |
| Runtime diagnostics | `curl -s http://127.0.0.1:9001/diag \| jq .` |
| Operator presets | `curl -s http://127.0.0.1:9001/operator/presets \| jq .` |

## Architecture At A Glance

The scheduler loads configuration, validates broker/data-feed settings, prepares
the runtime, fetches market data, computes signals, applies risk and execution
guards, submits broker orders through the Alpaca adapter, and exposes health,
diagnostics, and metrics through the Flask API.

Important boundaries:

- Broker runtime uses `alpaca-py==0.42.1`.
- Live mode fails closed for unsafe quote/provider combinations.
- Yahoo is for non-live historical fallback and research workflows only.
- Health endpoints must degrade gracefully and use the canonical shared helpers.
- Runtime code uses structured logging and must not add raw `print` calls.

For deeper design notes, read [ARCHITECTURE.md](ARCHITECTURE.md).

## Deployment

The supported production path is the packaged systemd workflow:

```bash
sudo cp packaging/systemd/ai-trading.service /etc/systemd/system/ai-trading.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-trading.service
sudo systemctl status ai-trading.service
curl -s http://127.0.0.1:9001/healthz | jq .
```

Use these deployment references:

- [DEPLOYING.md](DEPLOYING.md) - systemd deployment and required environment.
- [docs/DEPLOYING.md](docs/DEPLOYING.md) - detailed operational topology.
- [docs/OPERATIONS.md](docs/OPERATIONS.md) - runtime checks and operator APIs.
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - health, metrics, diagnostics,
  and operator endpoints.

## Documentation Map

- [AGENTS.md](AGENTS.md) - authoritative editing and runtime contract.
- [ARCHITECTURE.md](ARCHITECTURE.md) - runtime architecture.
- [DEPLOYING.md](DEPLOYING.md) - current production deployment.
- [docs/API_KEY_SETUP.md](docs/API_KEY_SETUP.md) - Alpaca credential setup.
- [docs/provider_configuration.md](docs/provider_configuration.md) - data-feed
  and fallback-provider policy.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - operator troubleshooting.
- [CHANGELOG.md](CHANGELOG.md) - notable changes.
- [SECURITY.md](SECURITY.md) - security reporting and trading-safety notes.

Root-level historical reports and implementation snapshots are archival unless a
current document links to them as authoritative.

## Optional Feature Sets

```bash
./venv/bin/python -m pip install "ai-trading-bot[plot]"
./venv/bin/python -m pip install "ai-trading-bot[ml]"
./venv/bin/python -m pip install "ai-trading-bot[ta]"
./venv/bin/python -m pip install "ai-trading-bot[train]"
```

TA-Lib and PyTorch may require platform-specific wheels or system libraries.
Install only the extras you need for the workflow you are running.

## Repository Metadata

Suggested GitHub About description:

> Python 3.12 Alpaca trading runtime with ML signals, risk controls, systemd
> deployment, health endpoints, and Prometheus metrics.

Suggested topics:

`algorithmic-trading`, `alpaca`, `python`, `machine-learning`, `trading-bot`,
`flask`, `prometheus`, `systemd`, `paper-trading`, `risk-management`.

## License

MIT. See [LICENSE](LICENSE).
