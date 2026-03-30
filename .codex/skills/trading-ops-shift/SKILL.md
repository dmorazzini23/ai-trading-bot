# Trading Ops Shift Skill

Use this skill when you want a single command for the current ops shift check
(pre-open, midday, or post-close).

## One-Command Run

```bash
cd /home/aiuser/ai-trading-bot
source venv/bin/activate
python scripts/ops_shift_check.py --phase auto | jq .
```

## Explicit Shift Runs

```bash
python scripts/ops_shift_check.py --phase pre_open | jq .
python scripts/ops_shift_check.py --phase midday | jq .
python scripts/ops_shift_check.py --phase post_close | jq .
```

## What It Checks

- `health_probe` + service state
- runtime KPI snapshot
- market-event risk window
- metrics backend/trend checks (midday)
- EOD summary readiness checks (post-close)

## Guardrails

- Read-only diagnostics only; no order placement.
- Restarts remain explicit and separate (`mcp_ops_server` or `mcp_infra_cloud_server`).
