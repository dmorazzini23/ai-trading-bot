# OpenClaw Hooks

These modules shape local webhook payloads into high-signal prompts for OpenClaw agents.

Configured local ingress:

- base URL: `http://127.0.0.1:18789/hooks/ai-trading-bot`
- routes:
  - `/github`
  - `/runtime`
  - `/coding`

The runtime connector builds the default OpenClaw target as
`/hooks/ai-trading-bot/runtime`. Override `AI_TRADING_OPENCLAW_GATEWAY_URL`,
`AI_TRADING_OPENCLAW_RUNTIME_WEBHOOK_URL`, or the OpenClaw config `hooks.path`
only when the local gateway route is intentionally different.

Research completion summaries are Slack-channel notifications routed to
`#all-beatwallstreet` by the packaged research units. Keep that Slack channel
separate from the OpenClaw runtime webhook route above; changing one does not
change the other.

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
