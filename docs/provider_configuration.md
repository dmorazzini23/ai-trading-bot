# Provider Configuration

This project supports multiple market data sources. Operators can toggle providers per environment using environment variables.

## Finnhub

- `ENABLE_FINNHUB`: set to `1` to enable, `0` to disable.
- `FINNHUB_API_KEY`: required when Finnhub access is enabled.

## Alpaca Feed

- `ALPACA_DATA_FEED`: choose `iex` (default) or `sip`. The `sip` option requires a SIP-enabled Alpaca account.

## Backup Provider

- `BACKUP_DATA_PROVIDER`: fallback source when the primary feed returns empty data. The default is `yahoo`. Set to `none` to disable backup queries.
- When a fallback is used, the bot logs `USING_BACKUP_PROVIDER` with the chosen provider. If disabled or unknown, `BACKUP_PROVIDER_DISABLED` or `UNKNOWN_BACKUP_PROVIDER` is logged.

Configure these variables in your deployment environment to control provider availability and failover behavior.
