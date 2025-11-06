Degraded Data Playbook

Purpose: ensure the bot behaves safely and predictably when Alpaca market data is unavailable or stale and the system falls back to backup/synthetic providers.

Key runtime toggles (set via environment)

- `EXECUTION_REQUIRE_REALTIME_NBBO=1`
  - Blocks opening new positions when quotes are degraded (non‑NBBO or stale).
- `TRADING__DEGRADED_FEED_MODE=block`
  - Explicitly blocks entries on degraded feeds (safe default if you prefer no trades over stale pricing).
- `EXECUTION_MARKET_ON_DEGRADED=1`
  - Opt‑in: if you still want to trade on degraded data, convert entries to market orders instead of placing potentially unfillable limit orders. Use only if you accept this risk.
- `EXECUTION_FALLBACK_LIMIT_BUFFER_BPS=0`
  - When using fallback pricing hints, disable extra widening and allow automatic downgrade to market (existing logic) instead of posting wide limits.
- `ALPACA_FALLBACK_TTL_SECONDS=30`
  - Shortens the decision window before retrying Alpaca as primary.
- `FALLBACK_TTL_SECONDS=60`
  - Separately reduces the in‑module minute‑data fallback TTL (affects repeated primary retries within the same cycle/window).
- `ALPACA_DATA_FEED=iex`
  - Prefer IEX on paper unless you have SIP entitlements. Do not set SIP unless authorized.

What the code already enforces

- Real‑time NBBO requirement is honored: when degraded and `EXECUTION_REQUIRE_REALTIME_NBBO=1`, entries are skipped with `ORDER_SKIPPED_PRICE_GATED` (reason `realtime_nbbo_required`).
- Opt‑in market on degraded: with `EXECUTION_MARKET_ON_DEGRADED=1`, the engine downgrades entries to market and logs `ORDER_DOWNGRADED_TO_MARKET` with provider and mode context.
- Correct order logging: `EXEC_ENGINE_EXECUTE_ORDER` includes the real `order_id`/`client_order_id` for traceability.
- Safe-mode minute-gap escalation now requires a confirmed Alpaca primary gap. The monitor ignores fallback-only payloads and emits `gap_metrics` (last/peak ratios, missing bars, event counts) when safe mode triggers so you can reconcile the halt.
- Yahoo fallback minute bars are automatically reindexed/interpolated to produce a contiguous series. Coverage metadata records `fallback_contiguous=True` so downstream gap-ratio checks treat the repaired frame as complete.
- When Alpaca quotes are unavailable but the fallback minute feed is contiguous, the trade gate synthesizes a quote timestamp from the repaired data. Orders are no longer rejected solely because the Alpaca quote API is offline, provided the fallback coverage remains healthy.

Suggested baseline for production safety

- `EXECUTION_REQUIRE_REALTIME_NBBO=1`
- `TRADING__DEGRADED_FEED_MODE=block`
- `ALPACA_FALLBACK_TTL_SECONDS=30`
- `FALLBACK_TTL_SECONDS=60`

Optional fill‑focused setup (accepting more risk)

- `EXECUTION_REQUIRE_REALTIME_NBBO=0`
- `EXECUTION_MARKET_ON_DEGRADED=1`
- `EXECUTION_FALLBACK_LIMIT_BUFFER_BPS=0`
- Keep shortened TTLs as above to recover to Alpaca faster.

Verification checklist

- Expect to see `ORDER_SKIPPED_PRICE_GATED` with `reason=realtime_nbbo_required` when degraded and NBBO gating is enabled.
- When market‑on‑degraded is enabled, expect `ORDER_DOWNGRADED_TO_MARKET` and no limit‑order slippage warnings for those entries.
- Portfolio should no longer remain perpetually at `positions=0` solely due to unfillable fallback‑priced limits.

