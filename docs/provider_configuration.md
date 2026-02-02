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
- `ALPACA_ALLOW_SIP` plus at least one of `ALPACA_HAS_SIP` or `ALPACA_SIP_ENTITLED` must resolve truthy for SIP requests to be issued.
  Setting `ALPACA_SIP_UNAUTHORIZED=1` forces the client back to IEX even when the other flags are present. `DATA_FEED_INTRADAY` and
  `ALPACA_DATA_FEED` honour these flags automatically—when SIP is not fully authorised the runtime silently downgrades to IEX.
- `ALPACA_FEED_FAILOVER`: comma-separated Alpaca feeds to try when a 200 OK response is empty. Example: `sip,iex` tells the bot to
  retry SIP first, then fall back to IEX if SIP is also empty or unavailable. Feeds that are not permitted by entitlement are
  ignored.
- `ALPACA_EMPTY_TO_BACKUP`: when set to `1`, the bot will jump directly to the configured backup provider if every preferred
  Alpaca feed responds with an empty payload.

When `ALPACA_DATA_FEED=iex`, the provider monitor automatically relaxes gap
detection thresholds. Minute safe-mode alerts now require a ~30% gap on IEX
data (SIP still uses the legacy 2% trigger), and the bot only aborts a cycle
for low minute coverage when the repaired frame drops below roughly 75% of the
intraday lookback window. The SIP feed retains its stricter tolerance so that
full-tape outages still halt trading promptly. See the
[Degraded Data Playbook](degraded_data_playbook.md) for operational guidance.

When a fallback feed returns usable data, the system records the switch per `(symbol, timeframe)` and logs `ALPACA_FEED_SWITCH`
once. Future requests for the same pair use the working feed immediately, eliminating redundant retries against an empty feed.

## Backup Provider

- `BACKUP_DATA_PROVIDER`: fallback source when the primary feed returns empty data. The default is `yahoo`. Set to `finnhub` to use the Finnhub low-latency candles API when an API key is configured, or to `none` to disable backup queries.
- When a fallback is used, the bot logs `USING_BACKUP_PROVIDER` with the chosen provider. If disabled or unknown, `BACKUP_PROVIDER_DISABLED` or `UNKNOWN_BACKUP_PROVIDER` is logged.
- Daily-bar requests automatically fall back to the configured provider when the primary response is missing required OHLCV columns, ensuring cached consumers always receive canonical data.

## Provider Priority and Fallbacks

- `DATA_PROVIDER_PRIORITY`: comma-separated order of providers to try. Default is `alpaca_iex,yahoo` unless SIP is entitled, in
  which case `alpaca_sip` is inserted after `alpaca_iex`.
- `MAX_DATA_FALLBACKS`: maximum number of fallbacks allowed before giving up. Default is `2` (tries both Alpaca feeds before Yahoo).

Configure these variables in your deployment environment to control provider availability and failover behavior.

## HTTP Host Limit Precedence

The per-host concurrency limit resolves environment knobs in the following order: `AI_TRADING_HTTP_HOST_LIMIT`,
`AI_TRADING_HOST_LIMIT`, `HTTP_MAX_PER_HOST`, and finally the legacy `AI_HTTP_HOST_LIMIT`. Updating any of these variables at
runtime triggers a semaphore refresh so both the data-fetch fallback workers and the shared HTTP pooling layer observe the new
limit immediately.

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
