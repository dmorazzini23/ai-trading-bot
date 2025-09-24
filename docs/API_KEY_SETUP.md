# API Key Configuration Guide

## Overview

This trading bot requires API keys from Alpaca Markets to function. This guide explains how to properly configure your API keys for both development and production use.

## 🔐 Security Best Practices

- **Never commit real API keys to version control**
- **Use paper trading keys for development and testing**
- **Use live trading keys only for production**
- **Store API keys in environment variables or secure .env files**
- **Regularly rotate your API keys**

## 📋 Required API Keys

### Primary (Required)
- `ALPACA_API_KEY`: Your Alpaca Markets API key
- `ALPACA_SECRET_KEY`: Your Alpaca Markets secret key
- `ALPACA_BASE_URL`: Alpaca API endpoint URL
- `WEBHOOK_SECRET`: Secret for webhook authentication
- Alternatively, set `ALPACA_OAUTH` with an OAuth token instead of the API key and secret. **Do not** set both.

> **Note:** The configuration loader trims surrounding whitespace and
> normalizes common environment prefixes (e.g. `DEV_ALPACA_API_KEY`) to
> their standard names, so you can keep environment-specific variables
> without additional code changes.

### Optional
- `FINNHUB_API_KEY`: For additional market data
- `NEWS_API_KEY`: For news sentiment analysis
- `SENTIMENT_API_KEY`/`SENTIMENT_API_URL`: For external sentiment service
- `FUNDAMENTAL_API_KEY`: For fundamental analysis data

## 🚀 Quick Setup

### 1. Get Your Alpaca API Keys

1. Go to [Alpaca Markets](https://app.alpaca.markets/paper/dashboard/overview)
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Generate new API keys:
   - **For Development**: Use Paper Trading keys
   - **For Production**: Use Live Trading keys

### 2. Configure Your .env File

Copy `.env.example` to `.env` and replace the sample values:

```bash
cp .env.example .env
```

Edit `.env` and replace these values:

```env
# Production environment configuration
# IMPORTANT: Replace these sample values with your real credentials
# Get your keys from: https://app.alpaca.markets/paper/dashboard/overview
# Provide ALPACA_API_KEY/ALPACA_SECRET_KEY or ALPACA_OAUTH (not both)
ALPACA_API_KEY=YOUR_ACTUAL_API_KEY_HERE
ALPACA_SECRET_KEY=YOUR_ACTUAL_SECRET_KEY_HERE
# ALPACA_OAUTH=YOUR_OAUTH_TOKEN_HERE
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional API keys - replace with your real keys or leave commented out
FINNHUB_API_KEY=YOUR_FINNHUB_API_KEY
NEWS_API_KEY=YOUR_NEWS_API_KEY
SENTIMENT_API_KEY=YOUR_SENTIMENT_API_KEY
SENTIMENT_API_URL=https://api.sentiment.example.com
WEBHOOK_SECRET=YOUR_STRONG_WEBHOOK_SECRET
```

### 3. API Key Formats

**Alpaca API keys follow specific formats:**
- API Key: Starts with `PK` for paper trading or `AKFZ` for live trading
- Secret Key: Starts with `SK` for paper trading or similar for live trading

Example formats (these are NOT real keys):
```
ALPACA_API_KEY=PKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ
ALPACA_SECRET_KEY=SKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD
```

## 🧪 Development vs Production

### Development/Testing
- Use **Paper Trading** keys
- Set `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
- These keys allow testing without real money

### Production
- Use **Live Trading** keys
- Set `ALPACA_BASE_URL=https://api.alpaca.markets`
- ⚠️ **WARNING**: These keys trade with real money!

### Position Reconciliation
- The bot reconciles local positions with the broker only when valid
  credentials are provided.
- In test or development environments, supply **paper trading** keys to
  enable reconciliation against the paper endpoint.
- If no broker client is configured, the bot logs a single warning and
  skips the reconciliation step.

## 🔧 Alternative Configuration Methods

### Environment Variables
Instead of using .env file, you can set environment variables directly:

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

### Docker Environment
For Docker deployments, use environment variables or Docker secrets:

```yaml
# docker-compose.yml
environment:
  - ALPACA_API_KEY=${ALPACA_API_KEY}
  - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
  - ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## ✅ Verification

To verify your configuration is working:

```bash
python -c "
import config
print('API Key configured:', bool(config.ALPACA_API_KEY and config.ALPACA_API_KEY != 'YOUR_ALPACA_API_KEY_HERE'))
print('Secret Key configured:', bool(config.ALPACA_SECRET_KEY and config.ALPACA_SECRET_KEY != 'YOUR_ALPACA_SECRET_KEY_HERE'))
print('Using paper trading:', 'paper' in config.ALPACA_BASE_URL.lower())
"
```

### Programmatic access

You can retrieve these credentials via a typed helper:

```py
from ai_trading.broker.alpaca_credentials import resolve_alpaca_credentials

creds = resolve_alpaca_credentials()
print(creds.api_key)
```

`resolve_alpaca_credentials` returns an `AlpacaCredentials` dataclass with
`api_key`, `secret_key`, and `base_url` fields along with `api_source` and
`secret_source` metadata describing which environment variables were used.

## ❌ Common Issues

### Issue: "Missing required environment variables"
**Solution**: Ensure all required API keys are set in your .env file or environment variables.

### Issue: "API key appears to be invalid"
**Solution**: 
- Check that your API key format is correct
- Verify you're using the right keys (paper vs live)
- Ensure there are no extra spaces or quotes

### Issue: "ALPACA_BASE_URL must start with http:// or https://"
**Solution**: Ensure your base URL includes the protocol (https://)

## 🔒 Security Notes

- **Never share your API keys publicly**
- **Don't commit .env files with real keys to git**
- **Use different keys for different environments**
- **Monitor your API key usage in Alpaca dashboard**
- **Revoke and regenerate keys if compromised**

## 📞 Support

If you need help:
1. Check the [Alpaca Documentation](https://alpaca.markets/docs/)
2. Review this project's logs for specific error messages
3. Ensure your account has the necessary permissions
4. Contact Alpaca support for account-specific issues
