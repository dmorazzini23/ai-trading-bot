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
- Slash commands exposed in Slack:
  `/openclaw`, `/help`, `/agentstatus`, `/model`, `/mode`, `/think`, `/new`, `/reset`, `/compact`
- Change session behavior from Slack:
  `/model codex/gpt-5.4-mini`
  `/model codex/gpt-5.4`
  `/mode`
  `/think off|minimal|low|medium|high`

## Hooks

- Base path: `http://127.0.0.1:18789/hooks/ai-trading-bot`
- Active transform directory: `~/.openclaw/hooks/transforms`
