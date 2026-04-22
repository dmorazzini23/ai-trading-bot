# ai-trading-bot heartbeat checklist

- Stay quiet unless there is drift, breakage, or a meaningfully useful summary.
- Check `ai-trading.service` health a few times per day, especially around market open and after any reported issue.
- If `/healthz` is unhealthy, inspect recent journal lines and summarize only the actionable issue.
- Periodically verify OpenClaw gateway and Slack connectivity.
- Surface hook, cron, or memory failures only when they are actionable.
- Avoid overnight noise unless something is broken or risky.
