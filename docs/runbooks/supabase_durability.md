# Supabase Durability Decision

Supabase is optional for runtime. The trading bot must remain able to operate
from the VPS durability stack without treating Supabase as live execution
authority.

## Recommended Role

Use Supabase for durable analytics and history:

- trade telemetry snapshots
- daily research and trading-day reports
- live-capital readiness history
- model promotion history
- live cost model history
- operator review notes

Do not use Supabase as the source of truth for live order submission, broker
state, quote authority, kill switches, or emergency stops.

## Keepalive Path

If the Supabase project sends inactivity warnings, keep it intentionally active
by writing low-risk analytics snapshots after daily research runs. A future
integration can upload these generated artifacts:

- `daily_readiness_latest.json`
- `trading_day_latest.json`
- `live_capital_readiness.json`
- `live_cost_model.json`
- promotion report metadata

The upload job should fail open for trading and fail closed for analytics
quality. A Supabase outage must not block exits or runtime health.

## Operator Policy

Before making Supabase authoritative for any new surface, write a migration plan
and a rollback plan. Until then, Supabase is useful evidence storage, not a live
capital control plane.
