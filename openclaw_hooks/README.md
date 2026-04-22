# OpenClaw Hooks

These modules shape local webhook payloads into high-signal prompts for OpenClaw agents.

Configured local ingress:

- base URL: `http://127.0.0.1:18789/hooks/ai-trading-bot`
- routes:
  - `/github`
  - `/runtime`
  - `/coding`

Authentication:

- bearer token required
- header options:
  - `Authorization: Bearer <token>`
  - or `x-openclaw-token: <token>`

Example local test:

```bash
curl -sS \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"source":"runtime","service":"ai-trading.service","severity":"warning","summary":"Sample local runtime check"}' \
  http://127.0.0.1:18789/hooks/ai-trading-bot/runtime
```
