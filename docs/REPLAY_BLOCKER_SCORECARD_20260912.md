# Replay and execution evidence scorecard — September 12, 2026

Decision: operational replay fixes are adopted by the production CLI; evidence
still does not qualify a model. No strategy search, training, threshold change,
fee substitution or order submission was performed. Holdout-overlapping replay
data was used for operational diagnostics only, not model selection.

## Scheduled adoption

The installed timer is active, next due September 14 at 09:22:18 UTC (includes
jitter). Its last service invocation was September 11 at 09:23:31 UTC, exit 2,
Result=success. The unit intentionally accepts exit 2 as a completed blocked
evaluation; this is not a qualification pass.

Manual `sudo -n systemctl start ai-trading-replay-governance.service` required an
unavailable password. Instead, as aiuser, loaded the unit's runtime environment
and ran its exact replay CLI with production output and summary paths. At
17:46:28 UTC it wrote the artifact and returned status=blocked with
REPLAY_POLICY_NON_REGRESSION_FAILED. Thus production CLI adoption is verified;
post-fix timer-triggered execution remains pending. No timer settings changed.

Artifact: `/var/lib/ai-trading-bot/runtime/replay_outputs/replay_hash_20260912.json`

SHA256: `227d76e7740c36bd64fb7ff07c1f2555962c01dea08cc3e141c001ba2bcc793e`

Summary: `/var/lib/ai-trading-bot/runtime/replay_governance_refresh_latest.json`.
Log: `/tmp/replay-next-production.log`. The summary is deliberately blocked.

## Timing alignment

Freshly read selected model metadata at
`/var/lib/ai-trading-bot/models/models/ml_edge-histgb-236ba0fe-20260717200350-4045d5c5/meta.json`:
one bar, required timeframe 5Min. Recorded label durations range from five minutes
to 89.6667 hours; their median is five minutes. A next available bar is not
necessarily the next five-minute interval.

| Candidate interval, seconds | Minimum | Median | P90 | Maximum | Exactly 300 | Above 300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision to fill | 46.245235 | 622.673643 | 10800 | 13444.838969 | 26 | 47 |
| Fill to markout | 60.158957 | 300 | 3900 | 81628.233246 | 38 | 33 |

P90 uses sorted index floor(0.9*(n-1)); n=77. The remaining six markouts are
shorter than 300 seconds. Baseline n=525: 286 markouts exactly 300 seconds, 228
longer, 11 shorter. These are descriptive coverage counts, not subgroup returns.
All candidate/baseline rows passed chronological and elapsed-time arithmetic
assertions. Candidate 77 usable plus 6 excluded reconciles to 83 fills; baseline
525 plus 14 reconciles to 539. Expiry repair does not make sparse next-observation
markouts a fixed-horizon strategy outcome. No horizon was changed or optimized.

## Execution-cost reconciliation

Refetched 90-day paper account activities at 17:46:10 UTC with complete pagination.
535 unique activities comprise 452 fills and 83 fees. 407 order quantities match
local evidence, but 407 local fee totals remain unknown. Another 6,989 local rows
were rejected as account-unverified; these are not silently joined to this account.
Quantity matching does not certify fill identity or complete costs.

None of the 452 fill activity rows contains fee_amount; none of the 83 fee rows
contains a fill/order reference. Account charges remain aggregated by reported
accounting date, never allocated to individual fills. This source cannot establish
complete per-fill fees. Simulation fee assumptions remain assumptions.

Report: `/tmp/replay-next-accounting.json`, SHA256
`6837be0ca4f3208a51a463a55611cb9b69ce3fa4d85a53139bb2188abc42637c`.
Raw private snapshot: `/tmp/replay-next-account-snapshot.json`, SHA256
`54227b48fbfad1e7ccc1c2db5b6b7a1a2f46a8caf5849dadc71d4f209d5c0a67`.
No account identifiers or raw transactions are copied into this document.

## Blockers and completion criteria

| Category | Finding | Completion criterion |
| --- | --- | --- |
| Operational verification | Production CLI adopted; timer run pending | After September 14 timer execution, verify new artifact fields, timestamp, summary and service result together |
| Measurement contract | Sparse next-observation replay differs from five-minute target | Specify decision, entry and exit timing under the governed protocol; demonstrate exact-horizon coverage and reject unavailable observations; no performance-based horizon selection |
| Source evidence | Per-fill costs unavailable from fetched activities | Obtain independently attributable complete execution fee evidence; preserve unknown values until then |
| Qualification failure | Candidate -9.750137 bps, 77 samples; baseline -7.181490 bps | Meet existing positive-edge, 250-sample and non-regression requirements with eligible evidence; never force trades or relax gates |
| Model eligibility | Selected model stale | A separately authorized, qualified candidate with valid provenance; refreshing a timestamp is insufficient and training remains paused |

No additional runtime defect was established by this audit. The next bounded
correctness task is the timing contract and coverage diagnostic; profitability
and missing external fee evidence cannot be resolved by declaring this audit passed.

## Validation and runtime

`pytest -q tests/test_replay_governance_tool.py tests/tools/test_broker_accounting_evidence.py`:
12 passed. Existing IOC/expiry validation remains applicable to unchanged code.
Fresh artifact assertions verify counts, timestamps and intervals. Non-sending
incident snapshot passed. Health at 17:47:07 UTC: broker connected/fresh, zero
positions/orders, no broker failures; existing replay and stale-model flags remain.
Only documentation changed in this task; no new behavior required new unit tests.
Docs-only validator and diff whitespace check are recorded in the handoff.
