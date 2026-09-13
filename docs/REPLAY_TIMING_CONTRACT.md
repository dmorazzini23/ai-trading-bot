# Replay timing contract

September 12, 2026. The replay summary now persists a versioned `timing_contract`
and `timing_diagnostics` in both candidate and baseline results, including empty
results. The existing qualification metric and thresholds remain unchanged.

The decision anchor is the simulated order submission timestamp, not certified
causal source-decision time. Entry is the simulated fill-event timestamp. The
existing metric exits at the first valid same-symbol observation strictly after
the fill, subject to its existing horizon ceiling. It is explicitly not a fixed
five-minute strategy return.

For each fill, diagnostics inspect exact observations 300 seconds after both
submission and fill. This fixed diagnostic interval corresponds to the audited
five-minute target; it is not a tunable performance parameter. No prices are
interpolated, carried forward, or selected by return. Duplicate identical prices
are accepted; conflicting prices at an exact timestamp are flagged. Missing or
invalid anchors remain explicit. Per-fill IDs, target timestamps, observation
statuses and entry timing statuses are saved; counts cover all fill events,
including those excluded from the existing markout metric.

These diagnostics do not calculate alternative returns, grant qualification, or
prove quote freshness, executable liquidity, source lineage, or training-label
alignment. Availability means a valid replay price exists at that exact timestamp.
Fills at or after the decision-plus-300-second target are identified separately.

## Operational verification

Isolated replay: `/tmp/timing-check/output/replay_hash_20260912.json`.
SHA256: `defb0930558001ca89dbe2dd2f3273540d932dccb15d45fa4ae3bd1495902c39`.
CLI summary: `/tmp/timing-check/summary.json`; exit 2,
`REPLAY_POLICY_NON_REGRESSION_FAILED`, as expected.

| Coverage | Candidate | Baseline |
| --- | ---: | ---: |
| All fill events | 83 | 539 |
| Exact decision-plus-300 observation available | 50 | 338 |
| Exact decision-plus-300 observation missing | 33 | 201 |
| Exact fill-plus-300 observation available | 40 | 290 |
| Exact fill-plus-300 observation missing | 43 | 249 |
| Fill at or after decision-plus-300 | 79 | 526 |

All diagnostic rows and category counts reconcile to fill counts. Source hashes
match the preceding production replay. Removing the two new diagnostic fields
leaves both summaries exactly equal to their prior production values. This
verifies no economic metric or exclusion changes. Holdout-overlapping data was
used only for operational contract checks, not strategy tuning or selection.

Regression tests cover exact observations beyond an earlier next observation,
missing target observations, duplicate/conflicting prices, invalid anchors,
late fills, excluded fills and empty summaries. Runtime health and non-sending
incident snapshot passed; the paper broker remained fresh with zero orders and
positions. Existing stale-model/replay flags remain. See CODEX_HANDOFF.md for
final validation results.

The scheduled replay CLI loads this checkout on each invocation; no main-service
restart is required for these replay-summary changes. Actual post-fix timer
verification remains due September 14. Rollback removes only timing-contract
additions in bot_engine.py and the new tests, preserving earlier expiry/IOC work.
