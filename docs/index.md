# AI Trading Bot Docs

## Core Runtime

- Main runtime entrypoint: `python3 -m ai_trading`
- Main API port: `9001`
- Default standalone health port: `8081`
- Canonical health route: `/healthz`
- Canonical config access: `ai_trading.config.management`

## Getting Started

Install and smoke-test:

```bash
python3 -m pip install -U pip
pip install -e .
python3 -m ai_trading --dry-run
ruff check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Run the standalone health app:

```bash
RUN_HEALTHCHECK=1 python3 -m ai_trading.app &
curl -s http://127.0.0.1:8081/healthz
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8081/metrics
```

Run the main API/runtime surface:

```bash
python3 -m ai_trading
curl -s http://127.0.0.1:9001/healthz
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9001/metrics
```

## Backtesting

The repository ships a deterministic CSV backtester module:

```bash
python3 -m ai_trading.strategies.backtester \
  --symbols SPY AAPL \
  --data-dir ./data \
  --start 2024-01-01 \
  --end 2024-12-31
```

## Deployment & Ops

- [Deployment Guide](DEPLOYING.md)
- [Operations](OPERATIONS.md)
- [Provider Configuration](provider_configuration.md)
- [Data Providers](data-providers.md)
- [SLO Alerts](slo_alerts.md)
- [Degraded Data Playbook](degraded_data_playbook.md)

## Strategy & Risk

- [Entry and Exit Signals](ENTRY_EXIT_SIGNALS.md)
- [Risk Engine](risk_engine.md)
- [Slippage](slippage.md)
- [Precision and Costs](precision_and_costs.md)

## Institutional Rollout

- [Institutional 30/60/90 Roadmap](institutional_roadmap_30_60_90.md)
- [Phase-Gated Roadmap](phase_gated_roadmap.md)
- [Acceptance Matrix](acceptance_matrix.md)
