# Data Provider Failover Flow

## Minute Feed Escalation

The bot now prioritises full-resolution sources before accepting delayed
providers when Alpaca gaps are detected. When continuity checks detect a
residual gap >2% or three or more missing minutes, the minute repair pipeline
promotes Finnhub via `ai_trading.data.fetch.fallback_order.promote_high_resolution`.
The promotion is cached for 15 minutes and forces `_backup_get_bars` to source
from Finnhub before Yahoo. Promotions are cleared automatically once Alpaca
recovers or when Finnhub returns empty data.

## Gap Diagnostics

Minute coverage metadata is persisted through
`ai_trading.data.fetch.validators.record_gap_statistics`. Tests can retrieve the
latest stats with `get_gap_statistics(symbol)` for assertions. The provider
monitor now suppresses transient alerts with configurable thresholds:

* `AI_TRADING_GAP_RATIO_SAFE_MODE` (default `0.02`) – minimum residual gap ratio.
* `AI_TRADING_GAP_MISSING_SAFE_MODE` (default `3`) – minimum missing bars.

When safe mode triggers, the monitor emits `PROVIDER_SAFE_MODE_DIAGNOSTICS` with
sample windows and counts to aid incident response.

## Quote Recovery

`ai_trading.execution.live_trading` recomputes quote timestamps whenever a
fallback quote is accepted. This unblocks the degraded-feed gate once a fresh
fallback quote (e.g. from Finnhub) is present, allowing trading to resume without
manual intervention.
