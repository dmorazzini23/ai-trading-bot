# OpenClaw Repo Memory

- Repo root is `/home/aiuser/ai-trading-bot`
- Prefer `./venv/bin/...` for Python tooling
- `ai-trading.service` is the live runtime to inspect, restart, and summarize
- `/healthz` is served on `http://127.0.0.1:9001/healthz`
- OpenClaw is loopback-only on `127.0.0.1:18789`
- Slack is the primary operator-facing surface
- Use `/model`, `/mode`, and `/think` to control the active OpenClaw model and reasoning level directly
