# API Key Configuration Guide

## Canonical Credentials

Use the current runtime config surface from `ai_trading.config.management`.
For Alpaca, the canonical trading endpoint variable is:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_TRADING_BASE_URL`

Optional, depending on your deployment:

- `ALPACA_OAUTH` instead of key/secret. Do not set both.
- `ALPACA_DATA_FEED` (`iex` or `sip`)
- `WEBHOOK_SECRET`
- `AI_TRADING_CAPITAL_CAP`
- `DOLLAR_RISK_LIMIT`

Deprecated and rejected by current startup validation:

- `ALPACA_API_URL`
- `ALPACA_BASE_URL`

## Recommended `.env`

```env
ALPACA_API_KEY=YOUR_ACTUAL_API_KEY
ALPACA_SECRET_KEY=YOUR_ACTUAL_SECRET_KEY
ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_FEED=iex
WEBHOOK_SECRET=YOUR_STRONG_WEBHOOK_SECRET
AI_TRADING_CAPITAL_CAP=0.25
DOLLAR_RISK_LIMIT=0.02
```

For live trading, switch only the trading base URL and your credentials:

```env
ALPACA_TRADING_BASE_URL=https://api.alpaca.markets
```

If your Alpaca account has SIP entitlements, you can enable SIP explicitly:

```env
ALPACA_DATA_FEED=sip
ALPACA_ALLOW_SIP=1
ALPACA_HAS_SIP=1
```

`DATA_FEED_INTRADAY` remains available for execution-pricing preferences, but
it does not replace the required `ALPACA_DATA_FEED` setting.

## Verification

Validate the resolved configuration:

```bash
python3 - <<'PY'
from ai_trading.config.management import validate_required_env

masked = validate_required_env()
print(masked)
PY
```

Inspect resolved Alpaca credentials:

```bash
python3 - <<'PY'
from ai_trading.broker.alpaca_credentials import resolve_alpaca_credentials_with_base

creds = resolve_alpaca_credentials_with_base()
print({
    "has_api_key": bool(creds.api_key),
    "has_secret_key": bool(creds.secret_key),
    "base_url": creds.base_url,
    "paper": "paper" in creds.base_url.lower(),
})
PY
```

## Common Issues

### Deprecated env key error

If startup reports that `ALPACA_API_URL` or `ALPACA_BASE_URL` is deprecated,
remove it and set `ALPACA_TRADING_BASE_URL` instead.

### Missing required environment variables

The current validation path requires more than just credentials. A minimal
runtime configuration also needs:

- `ALPACA_DATA_FEED`
- `WEBHOOK_SECRET`
- `AI_TRADING_CAPITAL_CAP`
- `DOLLAR_RISK_LIMIT`

### Invalid trading endpoint

`ALPACA_TRADING_BASE_URL` must be a full HTTP(S) URL such as:

- `https://paper-api.alpaca.markets`
- `https://api.alpaca.markets`

Do not use unresolved placeholders like `${ALPACA_HOST}` in `.env` or
`systemd` unit files unless you expand them before launch.
