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
- `ALPACA_FEED_FAILOVER`: comma-separated Alpaca feeds to try when a 200 OK response is empty. Example: `sip,iex` tells the bot to
  retry SIP first, then fall back to IEX if SIP is also empty or unavailable. Feeds that are not permitted by entitlement are
  ignored.
- `ALPACA_EMPTY_TO_BACKUP`: when set to `1`, the bot will jump directly to the configured backup provider if every preferred
  Alpaca feed responds with an empty payload.

When a fallback feed returns usable data, the system records the switch per `(symbol, timeframe)` and logs `ALPACA_FEED_SWITCH`
once. Future requests for the same pair use the working feed immediately, eliminating redundant retries against an empty feed.

## Backup Provider

- `BACKUP_DATA_PROVIDER`: fallback source when the primary feed returns empty data. The default is `yahoo`. Set to `finnhub` to use the Finnhub low-latency candles API when an API key is configured, or to `none` to disable backup queries.
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

### Tuning

Two environment variables control the backoff behaviour:

- `DATA_PROVIDER_BACKOFF_FACTOR`: multiplier applied to each successive
  disable. Defaults to `2`.
- `DATA_PROVIDER_MAX_COOLDOWN`: maximum cooldown in seconds before a provider
  is reconsidered. Defaults to `3600` (one hour).

When a provider recovers after being disabled, the monitor emits a
`DATA_PROVIDER_RECOVERED` log with the total outage duration and disable
frequency and raises a warning alert with the same metadata. These signals can
be scraped by external monitoring to surface provider flapping. On recovery,
the internal disable alert flag is cleared so subsequent outages trigger fresh
alerts.
