# Fee validation, session capture, and governed training

Implementation and checks: September 7, 2026.

## Changes

Execution comparisons now require an explicit fee contract: a finite,
nonnegative total per fill, USD currency, and a verified broker source for
observed fees. Missing fees, currency mismatches, cumulative order fees, and
configured estimates cannot complete net-cost validation. Simulated fills carry
their notional fee rate and amount through replay into the review; the amount
must match the declared calculation. Replay net edge and execution cost include
these fees when configured. The simulator's existing zero-fee default is now an
explicit research assumption, not a claim about broker fees.

Partial fills are aggregated by order. Slippage and total execution cost use
the same arrival benchmark, with each side's quantity as the denominator.
Thirty comparable orders across five sessions are required, and every paired
order must have a valid fee contract. `validated` means measurements are
comparable; it does not certify profitability or a small simulation error.
The report gives the measured mean net-cost error separately.

Live fill persistence preserves `fee_currency` and `fee_basis` from broker
evidence. It does not invent these fields when the broker payload omits them.
The required schema is `fee_amount`, `fee_currency: USD`,
`fee_basis: per_fill_total`, and `fee_source: broker_payload` or
`broker_activity`. Commission-only evidence must not be labeled total fees.

The scheduled paper review now exposes capture freshness, the next full
session's boundary windows, the actual boundaries used for reconciliation,
completion gaps, and the current training/selection results. The daily operator
summary carries these diagnostics. Existing broker synchronization records
position snapshots roughly once a minute, including outside market hours.

Accelerator CLI acquisition validation now resolves the governed dataset before
checking the legacy data directory. Daily, weekly, and weekend accelerator jobs
no longer skip a valid manifest because that legacy directory is missing.
Invalid manifests still block training; the resolved data directory also enters
the input signature. No freshness limit or promotion threshold was relaxed.

## Actual evidence and outstanding gates

Artifacts are in `artifacts/evidence_completion/`:

- `training/training_accelerator_report.json`: a bounded retrospective pipeline
  check fitted one hist-gradient, one-bar, risk-adjusted candidate on 147,169
  rows from the validated AAPL/AMZN/MSFT acquisition. This verifies the pipeline;
  it is not a new untouched research holdout. The run produced one named
  candidate and zero qualifying execution samples.
- `regime_selection.json`: the candidate reached selection, which retained the
  conservative fallback because development, sample, replay, and shadow
  requirements failed. Fresh training alone cannot replace the stale serving
  model. No model was promoted or serving registry modified.
- `governance_summary.json` and `replay/`: deterministic fixed-input replay
  retained baseline -17.4792 bps and candidate -17.9003 bps. The difference is
  within the configured non-regression tolerance; negative candidate net edge
  still blocks the overall governance gate.
- `paper_evidence_review.json`: actual capture was active, but the last completed
  session lacked complete boundaries and execution samples. The next full
  session is September 8, with opening capture 13:15-13:30 UTC and closing
  capture 20:00-20:15 UTC. No session or execution-cost completion is fabricated.
- `health_before.json` and `health_after.json`: live readiness remains HTTP 503
  because the required serving model is stale. Abstention remains enforced.

The installed daily research timer is active and next runs September 7 at
20:37 UTC. Its fresh subprocess will load the revised review and training code.
The paper service was restarted September 7 at 15:21 UTC to activate live fee
field persistence. Before restart it had zero open orders and positions.
Complete paper-position capture resumed at 15:22:37 UTC after restart.

## Validation and rollback

- `bash scripts/agent_validate_changed.sh --skip-runtime-smoke`: 725 tests
  passed, plus lint, type checks, compilation, and forbidden-pattern checks.
- The final public-CLI suite passed 14 tests; lint and mypy passed after adding
  training-result lineage assertions.
- Regression tests cover full review completion with explicit fees, missing and
  malformed fee contracts, partial fills, sell costs, fee propagation and
  deduction through replay, runtime fill-field persistence, calendar-aware
  capture readiness, governed training preflight, and scheduler wiring.
- Separate live-health checks were performed, and the non-sending incident
  snapshot check passed. No alert or order was sent by these checks.

Actual completion still needs a reconciled paper session, sufficient matched
executions with verified total fees, and a model satisfying existing eligibility
rules. Broker payloads without explicit total-fee evidence remain unknown.

Rollback: revert this task's fee, capture-diagnostic, and accelerator-preflight
changes and their tests, preserving earlier correctness fixes and evidence
artifacts. Restart the paper service if reverting live fee-field persistence.
