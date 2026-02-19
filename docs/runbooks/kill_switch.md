# Kill Switch Runbook

## Purpose

Emergency halt procedure for immediate stop of new risk and optional open-order cancellation.

## Activation

1. Set kill switch:

```bash
cd /home/aiuser/ai-trading-bot || exit 1
if grep -qE '^[[:space:]]*AI_TRADING_KILL_SWITCH=' .env; then
  sed -i -E 's|^[[:space:]]*AI_TRADING_KILL_SWITCH=.*|AI_TRADING_KILL_SWITCH=1|' .env
else
  printf '%s\n' 'AI_TRADING_KILL_SWITCH=1' >> .env
fi
sudo systemctl restart ai-trading.service
```

2. Confirm cancellation behavior is enabled:

- `AI_TRADING_CANCEL_ALL_ON_KILL=1`

## Verification

```bash
sudo journalctl -u ai-trading.service --since "10 min ago" --no-pager | \
grep -E 'KILL_SWITCH|CANCEL_ALL_TRIGGERED'
```

## Deactivation

1. Resolve root cause (data quality, broker outage, risk breach).
2. Set `AI_TRADING_KILL_SWITCH=0` and restart.
3. Validate normal decision/execution flow and absence of emergency triggers.

## Exit Criteria

- Root-cause incident closed.
- No active breaker/kill conditions.
- Normal provider and execution health restored.
