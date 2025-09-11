# Provider Configuration

This project supports multiple market data sources. Operators can toggle providers per environment using environment variables.

## Finnhub

- `ENABLE_FINNHUB`: set to `1` to enable, `0` to disable.
- `FINNHUB_API_KEY`: required when Finnhub access is enabled.

Example:

```bash
export FINNHUB_API_KEY=your_finnhub_key
export ENABLE_FINNHUB=1
```

## Alpaca Feed

- `ALPACA_DATA_FEED`: choose `iex` (default) or `sip`. The `sip` option requires a SIP-enabled Alpaca account.

## Backup Provider

- `BACKUP_DATA_PROVIDER`: fallback source when the primary feed returns empty data. The default is `yahoo`. Set to `none` to disable backup queries.
- When a fallback is used, the bot logs `USING_BACKUP_PROVIDER` with the chosen provider. If disabled or unknown, `BACKUP_PROVIDER_DISABLED` or `UNKNOWN_BACKUP_PROVIDER` is logged.

## Provider Priority and Fallbacks

- `DATA_PROVIDER_PRIORITY`: comma-separated order of providers to try. Default is `alpaca_iex,alpaca_sip,yahoo`.
- `MAX_DATA_FALLBACKS`: maximum number of fallbacks allowed before giving up. Default is `2` (tries both Alpaca feeds before Yahoo).

Configure these variables in your deployment environment to control provider availability and failover behavior.

## Adaptive Disabling & Switchover Monitoring

When Alpaca repeatedly returns empty data or errors, the bot now applies an
exponential backoff when disabling the provider. Each consecutive disable
period doubles, up to one hour, to avoid rapid flip/flop cycles. Whenever the
system falls back to another provider, a `DATA_PROVIDER_SWITCHOVER` log entry
is emitted with the running count for that provider pair to aid diagnostics.
