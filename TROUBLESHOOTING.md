# Troubleshooting Guide

## Overview

This guide covers common issues and debugging steps for the AI Trading Bot. The packaged service logs to journald, so use `sudo journalctl -u ai-trading.service` for logs. There is no `logs/scheduler.log`.

## Table of Contents

- [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
- [Common Issues](#common-issues)
- [Unresolved Environment Placeholders](#unresolved-environment-placeholders)
- [Debugging Procedures](#debugging-procedures)
- [HTTP Status Codes](#http-status-codes)
- [API Connectivity Problems](#api-connectivity-problems)

## Quick Diagnostic Checklist

### Initial Health Check

Run this checklist before diving into detailed troubleshooting:

```bash
# 1. Check service status
sudo systemctl status ai-trading.service

# 2. Verify configuration
./venv/bin/python -m ai_trading.tools.env_validate
curl -s http://127.0.0.1:9001/diag | jq .

# 3. Check health (packaged service serves /healthz on :9001)
curl -sS http://127.0.0.1:9001/healthz | jq .

# 4. Check logs for errors
sudo journalctl -u ai-trading.service -n 200 --no-pager | grep -i error

# 5. Verify Python environment
./venv/bin/python --version  # Python 3.12.x

# 6. Check system resources
df -h  # Disk space
free -h  # Memory usage
```

### Log Analysis Commands

```bash
# Recent errors
sudo journalctl -u ai-trading.service --since "1 hour ago" --no-pager | grep -i error

# Data fetch problems
sudo journalctl -u ai-trading.service --since "1 hour ago" --no-pager | grep -E "ALPACA_FETCH|DATA_WINDOW_NO_SESSION|ALPACA_IEX_FALLBACK_SIP"

# Degraded-feed entry blocks
sudo journalctl -u ai-trading.service --since "1 hour ago" --no-pager | grep DEGRADED_FEED_BLOCK_ENTRY
```

## Common Issues

### 1. Bot Won't Start

**Symptoms:**
- Service fails to start
- Immediate exit after launch
- ImportError or ModuleNotFoundError

**Solutions:**

```bash
# Check why the service stopped
sudo journalctl -u ai-trading.service -n 100 --no-pager

# Check Python version (requires >=3.12,<3.13)
./venv/bin/python --version

# Reinstall dependencies
make install-dev

# Check for missing environment variables (reports keys, never values)
./venv/bin/python -m ai_trading.tools.env_validate
```

Packaged services copy `/home/aiuser/ai-trading-bot/.env` to `/run/ai-trading-bot/ai-trading-runtime.env` on start. After editing `.env`, restart the service to refresh it.

### Module attribute errors

**Error:** `module 'ai_trading.config.management' has no attribute 'get_env'` (or `reload_env`).

**Fix:** Update to the latest code and ensure `ai_trading/config/management.py` exports both helpers.

### Unresolved Environment Placeholders

**Symptoms:**

- Runtime exit with `Missing required environment variable: ALPACA_TRADING_BASE_URL`.
- Config errors mentioning `ALPACA_API_URL` or `ALPACA_BASE_URL`, which are deprecated and not supported. Set `ALPACA_TRADING_BASE_URL` instead.
- A value that "looks like an unresolved placeholder" is rejected by the config loader.

**Why it happens:**

- The config loader rejects values that contain unresolved `${VAR}` placeholders.
- `systemd` does **not** expand shell-style placeholders in `Environment=` or `EnvironmentFile=` entries. If you ship a line such as `ALPACA_TRADING_BASE_URL=${ALPACA_HOST}`, the literal string `${ALPACA_HOST}` is passed to the bot.
- `.env` files read by `python-dotenv` also treat `${VAR}` literally unless you pre-render them.

**Fixes:**

1. Define the full endpoint explicitly, e.g. `ALPACA_TRADING_BASE_URL=https://api.alpaca.markets` (production) or `https://paper-api.alpaca.markets` (paper trading).
2. For the packaged service, edit `.env` and restart: `sudo systemctl restart ai-trading.service`. Do not add a second env source under `/etc/ai-trading-bot`.
3. Run `./venv/bin/python -m ai_trading.tools.env_validate` to confirm the environment passes validation.

**Verification:**

- `sudo journalctl -u ai-trading.service | grep ALPACA_TRADING_BASE_URL`
- `grep -c '\${' /run/ai-trading-bot/ai-trading-runtime.env` (a count above zero means placeholders remain; do not print the file, it holds secrets)

### Health endpoint returns 500

**Symptoms:** `curl -sS http://127.0.0.1:9001/healthz` shows 500 or non-JSON.

**Fix:** Look for the traceback in `sudo journalctl -u ai-trading.service`. Health routes are registered through the shared canonical helpers and are expected to degrade gracefully rather than raise.

### 2. API Connection Failures

**Symptoms:**
- "Connection refused" errors
- "Authentication failed" messages
- Timeouts during data fetching

See [API Connectivity Problems](#api-connectivity-problems) below.

### 3. Data Fetching Problems

**Symptoms:**
- Empty dataframes
- Stale data warnings
- Data provider fallback messages
- `ALPACA_EMPTY_BAR_MAX_RETRIES` in logs
- `ALPACA_FETCH_ABORTED` in logs
- `ALPACA_FETCH_RETRY_LIMIT` in logs

Repeated empty 200-responses trigger these limits quickly. Verify the market is
open or that data exists for the requested window (symbol still listed, feed
correct) before retrying. If the requested window contains no trading session,
the fetcher logs `DATA_WINDOW_NO_SESSION`.

Per-request retries (default **5**) can be tuned via the
`FETCH_BARS_MAX_RETRIES` environment variable. Yahoo HTTP fallback is enabled
by default; set `ENABLE_HTTP_FALLBACK=0` (or `false`) to opt out when running
in an offline environment. Fallbacks to a backup provider are counted in the
`BACKUP_PROVIDER_USED` metric (`ai_trading/data/fetch/metrics.py`).

When Alpaca's IEX feed returns empty responses, the fetcher can retry the same
request on the SIP feed before falling back to Yahoo Finance. This is logged
as `ALPACA_IEX_FALLBACK_SIP`.

#### Degraded-feed policy

- `TRADING__DEGRADED_FEED_MODE` selects the policy when market data is degraded:
  `block` (default; block entries), `widen` (widen protective limits and allow
  fallback quotes), or `hard_block` (hard block the cycle). See the
  `degraded_feed_mode` spec in `ai_trading/config/runtime.py`.
- In `block` mode, a blocked entry logs `DEGRADED_FEED_BLOCK_ENTRY` with the
  symbol, side, provider and quote age.
- `SKIP_COMPUTE_WHEN_PROVIDER_DISABLED` is a separate setting; check its
  description in `ai_trading/config/runtime.py` before relying on it.

### 4. Trade Execution Failures

**Symptoms:**
- Orders rejected by broker
- Insufficient buying power errors
- Position size calculation errors

Check the service journal for the order and rejection reason first
(`sudo journalctl -u ai-trading.service | grep -i reject`). Account status,
buying power and open orders can be checked with the Alpaca curl calls under
[API Connectivity Problems](#api-connectivity-problems).

## Debugging Procedures

### Enable Debug Logging

Set `LOG_LEVEL=DEBUG` in `.env` and restart the service
(`sudo systemctl restart ai-trading.service`), then read the journal. Revert
it when finished.

## HTTP Status Codes

| Code | Description | Action |
|------|-------------|--------|
| 400 | Bad Request | Check request parameters |
| 401 | Unauthorized | Verify API credentials |
| 403 | Forbidden | Check API permissions |
| 404 | Not Found | Verify endpoint URL |
| 422 | Unprocessable Entity | Validate request data |
| 429 | Too Many Requests | Back off and retry |
| 500 | Internal Server Error | Check server logs |
| 502 | Bad Gateway | Check upstream services |
| 503 | Service Unavailable | Wait and retry |

## API Connectivity Problems

### Alpaca API Issues

**Connection Testing:**

```bash
# Test Alpaca API connectivity (do not paste keys into shared logs)
curl -X GET \
  -H "APCA-API-KEY-ID: ${ALPACA_API_KEY}" \
  -H "APCA-API-SECRET-KEY: ${ALPACA_SECRET_KEY}" \
  "${ALPACA_TRADING_BASE_URL}/v2/account"
```

**Network Connectivity:**

```bash
# Test specific endpoints
curl -I https://api.alpaca.markets/v2/account
curl -I https://finnhub.io/api/v1/quote

# Check DNS resolution
nslookup api.alpaca.markets
nslookup finnhub.io
```
