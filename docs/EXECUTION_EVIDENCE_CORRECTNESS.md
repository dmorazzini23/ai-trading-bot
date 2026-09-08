# Execution evidence correctness follow-up

Verified September 7, 2026.

## Changes

- Session reconciliation counts missing or invalid fill timestamps as
  `unclassifiable_fill_timestamps` and blocks a clean session result. Since those
  rows cannot be assigned to a session, the count covers the retained fill input.
- The simulated broker requires a finite, positive observed market price for
  fills. Orders remain pending without one; order prices and the former $100
  fallback cannot manufacture executions, including during replay's final drain.
- Daily review reports explicit `completion_gaps`. A reconciled session and a
  supported slippage comparison cannot produce `review_complete` while net-cost
  validation lacks simulated and observed fee contracts.

The existing scheduled daily workflow invokes these canonical modules in a fresh
process. No service restart, model promotion, or trading-limit change was needed.

## Regression coverage and validation

- Missing, empty, and malformed timestamps block otherwise reconciled sessions.
- Missing, zero, negative, nonfinite, and malformed market prices leave orders
  pending; a later observed price fills once. Replay cannot invent trailing fills.
- A public CLI test supplies 30 matched fills across five sessions and a reconciled
  final session, then confirms missing fee contracts keep the review pending.
- `bash scripts/agent_validate_changed.sh --skip-runtime-smoke`: lint, type and
  compile checks passed; 666 targeted tests passed.
- Separate final daily-review suite: 6 passed; type and compile checks passed.
- Read-only live health and a non-sending incident snapshot were run separately.
  The snapshot check passed. Health returned HTTP 503 with
  `required_model_stale`; broker connectivity was healthy.

## Current runtime evidence

Artifacts are under `artifacts/three_correctness_fixes/`:

- `governance_summary.json` and `replay/replay_hash_20260907.json`: frozen-input
  replay consumed 4,349 source rows and produced 3,040 fill events. Determinism
  and OMS invariants passed. Candidate net edge was -17.9003 bps versus baseline
  -17.4792 bps. The -0.4211 bps difference passes the configured 0.5 bps
  non-regression tolerance, but negative candidate net edge still blocks governance.
- `paper_evidence_review.json`: `evidence_pending`, with session reconciliation,
  execution-comparison support, and net-cost validation gaps. Current input has
  zero unclassifiable fill timestamps. Complete session boundaries and execution
  samples are still missing; no successful session is claimed.
- `health.json`: captured live health response. This is runtime readiness evidence,
  not a claim that a new strategy is ready for promotion.

The remaining work requires valid execution/session evidence, explicit fee
contracts, and an eligible model. These fixes do not establish profitability.
Rollback consists of reverting this patch's three module changes and their
regression tests; reverting would restore the identified false-success behavior.
