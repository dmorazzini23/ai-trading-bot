# Operational readiness — September 7, 2026

The supported decision is **abstain from the required model's new exposure**.
No production fallback exists, the selected shadow model is stale, and the
corrected replay still fails its economic gate. Do not relax freshness, cost,
or replay requirements to make readiness green.

## Replay diagnosis and correction

The preserved September 6 runtime replay contains 776 limit-fill observations
that violate the order's limit, out of 808 baseline limit observations. The
simulator applied spread/volatility slippage without enforcing limit prices.
Its shared random stream also changed retained orders' fill draws when caps
removed unrelated orders. The corrected simulator waits for a compatible
observed market price, bounds execution to the limit, timestamps fills at the
observation used, and seeds separate random streams by client-order identity.
This remains a simplified simulator, without queue or market-impact evidence.

The comparison reran exactly 4,349 source rows with seed 42, a $25,000 symbol
cap and $150,000 gross cap. Both before and after source hashes are
`ea9716ffc2369596d2ba5ce240751035e978216eac0b368a5f94db67f21a3cbc`.
There are now zero through-limit fills among 780 baseline limit observations.
Replay determinism and OMS invariants pass.

| Corrected capped-minus-uncapped component | Contribution, bps |
|---|---:|
| Entry decisions | 0.000 |
| Order set retained by sizing/caps | -1.696 |
| Fill mix/timing | 0.000 |
| Execution-price cost difference | 0.000 |
| Exits | Not identifiable from this markout metric |
| Reconciliation residual | 0.000 |

Baseline mean markout is -16.827 bps; capped candidate is -18.523 bps.
There are 1,922 cap adjustments, including 1,139 fully blocked intents.
MSFT contributes -1.343 bps of the -1.696 bps difference. Caps also reduce
absolute quantity-weighted marked losses from $37,587 to $18,418. This is
marked accounting, not realized portfolio P&L; it does not justify removing
risk caps. The non-regression and positive-edge gates correctly remain failed.
The preserved earlier research run and latest runtime run use different source
sets; their headline results must not be combined as an identical-input test.

Evidence: `artifacts/operational_readiness/governance_summary.json` and
`artifacts/operational_readiness/replay/replay_hash_20260906.json`.

## Execution capture and reconciliation

The live execution path now retains account/mode identity, decision identity,
execution timestamps, fee provenance, and collision-resistant derived fill
identities when a broker execution ID is absent. Broker account identity is
retained across cycle-cache resets. Order events carry account/mode identity.
Successful broker position responses append complete boundary records,
including an explicitly empty portfolio; missing responses and malformed
positions cannot certify an empty portfolio. Boundary observations are not
atomic with the fill stream, and unknown account identity remains explicit.

`AI_TRADING_BROKER_POSITION_BOUNDARIES_PATH` defaults to
`runtime/broker_position_boundaries.jsonl`. Capture uses the existing
`AI_TRADING_RUNTIME_EXEC_EVENT_PERSIST_ENABLED` control. Estimated or legacy
unverified fees cannot certify net P&L. A zero fee is known only when actually
reported with broker provenance. No historical fees are manufactured.

The reconciliation CLI supports `--session YYYY-MM-DD --account-id ACCOUNT
--position-boundaries PATH`, alongside its required decisions, orders, fills,
TCA and output paths. It requires paper-account boundaries within 15 minutes
before the exchange open and after its close, respects early closes, and
checks fills over the entire boundary interval. Missing boundaries, lineage,
fees, TCA, quantity agreement, account identity, or execution samples keep the
session incomplete. The complete-session synthetic regression fixture passes;
this is pipeline validation, not a completed real paper session.

The historical audit still finds 7,384 valid fills, 7,107 without a retained
decision match, 1,267 without TCA, and 7,384 without verified fees. Time-aligned
historical opening/closing position snapshots are absent. These missing
records cannot be repaired by changing the reader.

Evidence: `artifacts/operational_readiness/execution_reconciliation.json`.
A complete observed paper-session audit remains pending new captured evidence.
An abstaining session with no fills must remain `no_execution_samples`; do not
place trades solely to obtain a passing audit.

## Governed serving decision

The runtime rich registry contains 187 `ml_edge` shadow entries and no
production entry. The newest shadow entry was trained July 17, 2026, more than
51 days ago versus the 14-day freshness requirement. It cannot serve new
model exposure. Readiness now states `serving_decision: abstain` and
`model_new_exposure_allowed: false`, matching the loader's existing rejection.
No model was promoted, demoted, relabeled, or automatically substituted.

Evidence: `artifacts/operational_readiness/model_serving_decision.json` includes
the selected identity, freshness limits and registry hash. Service health
continues to report `required_model_stale` and `replay_live_parity_gate_failed`;
broker connectivity and database checks pass. Risk-reducing order authority
is separate from the unavailable model's new-exposure decision.

## Validation and handoff

The first required changed-file validation passed 945 tests, lint, types,
compilation and forbidden-pattern checks. Additional capture-path checks
passed 25 tests. Final validation output is retained in
`/tmp/readiness-validation-final.log`. Tests cover limit reachability and
price bounds, random-draw independence, fill timestamps, fee provenance,
decision lineage, empty/malformed position boundaries, complete paper-session
reconciliation, and stale-model abstention.

The read-only health response is in `artifacts/operational_readiness/health.json`.
HTTP 503 is an explicit unmet-governance result, not a passed health check.
The future research holdout remains reserved.

The paper service was restarted on September 7 to activate the validated
changes. It is active and running, and
`artifacts/operational_readiness/health_after_activation.json` confirms the
explicit abstention decision, healthy broker connectivity, zero open orders,
and zero positions. The two governance attention flags remain. New records in
`/var/lib/ai-trading-bot/runtime/broker_position_boundaries.jsonl` contain the
paper account identity and complete empty-position snapshots. Capture is
therefore verified in the running service, not merely enabled in source.

Final mapped validation again passed 945 tests in 171 seconds. The final small
account-capture and invalid-fill checks were followed by 34 targeted tests,
eight reconciliation tests, lint/compile checks and a two-file mypy pass.
The non-sending incident snapshot check passed; no alerts were sent.

The last completed session audit is retained as
`artifacts/operational_readiness/last_paper_session.json`. It cannot establish
a complete observed session retroactively. Re-run that CLI for a subsequent
session once both contemporaneous boundaries and execution evidence exist.

Rollback: revert this task's simulator, execution capture, reconciliation and
model-readiness edits, retaining reports and earlier task changes. A runtime
restart is required to load or roll back Python execution-path changes.
