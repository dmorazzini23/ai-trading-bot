# Tasks for Agents (Live Backlog)

## Ready
- Ensure all hot paths accept `runtime` and do not reference a global `ctx`.
- Centralize ML model loading in `_load_primary_model(runtime)` and cache at `runtime.model`.
- Normalize metrics imports under `ai_trading.monitoring.metrics`.

## Next PR Bundle (already planned)
- Strategy re‑tuning and exception hygiene across modules.
- Broker migration / multi‑broker routing.
- ML architecture overhaul (beyond thresholding/ensemble gating).
- New data providers / alt‑data integrations.

## Non-Goals (for this repo pass)
- Destructive refactors or replacement of critical modules without explicit approval.
- Changing strategy logic unless the PR states so.
