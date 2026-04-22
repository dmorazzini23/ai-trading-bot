# TOOLS.md - ai-trading-bot local notes

## Repo

- Root: `/home/aiuser/ai-trading-bot`
- Python venv: `/home/aiuser/ai-trading-bot/venv`
- Preferred Python invocation: `./venv/bin/python`

## Services

- Trading service: `ai-trading.service`
- OpenClaw gateway: `openclaw-gateway.service`

## Commands

- Health:
  `curl -sS http://127.0.0.1:9001/healthz`
- Trading service status:
  `systemctl status ai-trading.service --no-pager`
- Trading service logs:
  `journalctl -u ai-trading.service -n 50 --no-pager`
- OpenClaw gateway status:
  `openclaw gateway status --token $(cat ~/.openclaw/gateway.token)`

## Narrow sudo for aiuser

- `systemctl status ai-trading.service`
- `systemctl restart ai-trading.service`
- `systemctl start ai-trading.service`
- `systemctl stop ai-trading.service`
- `systemctl show ai-trading.service`

## Slack

- Primary DM destination: `D0AUCEGTFGV`
- Paired user id: `U0900J4TTB9`
- Existing slash command: `/openclaw`
- Spawn a lane from Slack:
  `/openclaw /subagents spawn ops <task>`
  `/openclaw /subagents spawn coder <task>`
  `/openclaw /subagents spawn auditor <task>`
  `/openclaw /subagents spawn deep-coder <task>`
- Keep a thread on one lane:
  `/openclaw /agents`
  `/openclaw /focus <target>`
- Change session behavior from Slack:
  `/openclaw /model codex/gpt-5.4`
  `/openclaw /think high`
  `/openclaw /think xhigh`

## Hooks

- Base path: `http://127.0.0.1:18789/hooks/ai-trading-bot`
- Active transform directory: `~/.openclaw/hooks/transforms`
