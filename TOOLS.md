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
  `ai-trading-local-check health`
- Trading service status:
  `ai-trading-local-check service`
- Trading service logs:
  `ai-trading-local-check logs 50`
- OpenClaw gateway status:
  `openclaw-admin gateway status`

## Local loopback checks

- Do not use `web_fetch` for `http://127.0.0.1:*`, `http://localhost:*`, or other private/internal service URLs; OpenClaw's SSRF guard blocks those requests by design.
- Use local shell commands instead:
  - `ai-trading-local-check health`
  - `ai-trading-local-check triage`
  - `curl -fsS http://127.0.0.1:9001/healthz`

## Conversational routing

- Casual trading-status questions in Slack should trigger live checks automatically.
- Preferred evidence order for `How is trading going?` style questions:
  1. `/triage`
  2. `/service status`
  3. `ai-trading-local-check health`
  4. `ai-trading-local-check logs 50`
  5. `/runtime-report today`
- Reply with a short summary first, then the key blocker or healthy-state note.

## Slack reply semantics

- For the current Slack DM thread with Dom, plain assistant output is automatically delivered by OpenClaw.
- Do not call the Slack tool to send the same primary reply back into the same DM.
- Use the Slack tool only when the task itself is a Slack operation beyond the normal reply path.

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
