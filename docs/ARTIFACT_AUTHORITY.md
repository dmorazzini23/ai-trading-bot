# Artifact Authority

This document defines which artifacts are authoritative for operators and agents.
It prevents old research outputs from being mistaken for live-capital evidence.

## Authoritative Runtime Evidence

Use these latest pointers for current operational decisions:

- Runtime health: `http://127.0.0.1:9001/healthz`
- Daily research answer:
  `/var/lib/ai-trading-bot/runtime/research_reports/latest/daily_readiness_latest.json`
  is the canonical daily operator answer. The automation also writes
  `daily_research_latest.json` as a compatibility alias for older readers, but
  new operator workflows should read `daily_readiness_latest.json`.
- Trading-day attribution:
  `/var/lib/ai-trading-bot/runtime/research_reports/latest/trading_day_latest.json`
- Live-capital readiness:
  `/var/lib/ai-trading-bot/runtime/live_capital_readiness_latest.json`. The
  latest daily bundle also keeps its run-local `live_capital_readiness.json`,
  but the stable runtime pointer is the authority.
- Launch-profile runtime gate state:
  `/var/lib/ai-trading-bot/runtime/live_canary_state_latest.json` for
  `live_canary`, and `/var/lib/ai-trading-bot/runtime/launch_profile_state_latest.json`
  for other enforced launch profiles. These artifacts are submit-time guard
  evidence, not permission to bypass live-capital readiness.
- Live cost model:
  `/var/lib/ai-trading-bot/runtime/live_cost_model_latest.json`, or the current
  daily bundle's `live_cost_model.json`
- Replay governance:
  `/var/lib/ai-trading-bot/runtime/replay_outputs/` plus the health payload's
  `replay_live_parity_gate`
- Promotion reports:
  `artifacts/promotion/promotion_report_latest.json` when generated manually
- Training accelerator:
  `/var/lib/ai-trading-bot/runtime/training_accelerator_<cadence>_latest.json`
  for cached candidate research status. These reports have no promotion
  authority.
- Research automation summary:
  `/var/lib/ai-trading-bot/runtime/research_reports/latest/daily_operator_summary.json`
  plus
  `/var/lib/ai-trading-bot/runtime/research_reports/latest/daily_research_automation_latest.json`.
  Completion notifications must use the automation latest pointer for run
  status and must not substitute stale daily readiness aliases after failed or
  locked launches.

## Research Artifacts

Daily, weekly, and monthly research bundles are evidence, not authority to mutate
the runtime. They may contain candidate models, replay studies, symbol rankings,
or suggested gates. A candidate becomes operational only after a manual promotion
report and an operator cutover.

## Archive Rules

Old experimental outputs should move under `artifacts/archive/` or
`docs/archive/` when they are no longer referenced by tests or runbooks. Do not
delete historical evidence blindly. Do not move fixtures used by tests.

Agents should treat root-level `*_SUMMARY.md`, `*_REPORT.md`, and snapshot-style
documents as archival unless they are explicitly refreshed in the current task.

## Authority Rules

- Runtime health and live-capital readiness decide whether live money is even
  eligible.
- Promotion reports decide whether a model candidate is eligible for manual
  cutover.
- Daily research reports explain tomorrow's recommended mode and blockers.
- Trading-day reports explain what happened today.
- Supabase, if enabled, is durable analytics/history only. It is not live
  execution authority.
- RL and advanced models remain research/shadow unless a promotion report and
  live-readiness gate explicitly allow them.
