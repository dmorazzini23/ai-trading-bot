# October 2 EOD exit accounting repair

## 1. AMZN exit lineage and TCA

**Root cause:** the EOD flatten helper submitted directly through the canonical
execution engine. It bypassed the ordinary decision recorder and pending TCA
writer. The helper supplied a real trigger time, but the durable lifecycle writer
discarded it and used claim time. October 2's AMZN sell therefore has an intent,
claim, broker acknowledgement and captured fill, but no decision trace or TCA
receipt. Its intent account matches the observed paper broker; the account-chain
diagnostic failed because the matching decision is absent, not because a
different account was observed.

**Changes:** keep the existing session/symbol/side broker idempotency key and
protective reduction path. Add a stable exit trace, retain the original trigger
time/basis, and append an operational EXIT decision before a new durable claim.
Existing intents are never backfilled. Audit failures remain visible and do not
disable protective exits or relax submission gates. Explicit decision retries
query the original durable row so an interrupted OMS emission references its
original UUID, timestamp and context.

**Pre-deployment follow-up:** the actual fill-capture writer can attach the
position's existing entry correlation. The prospective EOD receipt must retain
that explicit correlation from exit metadata because the TCA reconciler requires
agreement before matching order IDs. Four known-correlation cases reproduced the
omission. The receipt now retains the known value; unknown identity stays unknown.
The expanded eight-case integration invokes the real fill-capture writer and
checks fill/receipt account, decision trace and position-entry correlation.
Its final required changed-file validation passed 57 mapped tests, Ruff,
mypy/compilation on both changed paths, health and the non-sending snapshot.
Log: `/tmp/amzn-exit-correlation-final-validation.log`. The earlier required
validation is reused for unchanged OMS/TCA/core paths; final exact-tip CI is
required for this follow-up before deployment.

The acknowledged EOD order now receives a prospective TCA request receipt.
Observed fills can resolve it without an invented arrival benchmark, fee total,
slippage or latency. Partial fills may advance with newer cumulative quantities;
duplicate and stale updates are rejected and earlier records remain immutable.
These operational receipts are ineligible for promotion and contain no cost
metrics usable by the live cost model. Unknown fees stay `null`; even a supplied
fee amount does not certify the complete per-fill total.

**Regression coverage:** the real flatten -> execute_order -> OrderManager ->
IntentStore/EventStore path for long and short reductions, known/unknown account,
submission ordering, causal time/trace, fill resolution and repeated requests.
Additional cases cover interrupted claims, immutable historical intents,
interrupted OMS decision emission, duplicate/stale partial fills, unknown
benchmarks, missing fill timestamps and unknown versus supplied fees.

**Validation:** 136 focused tests passed; two durable-claim retry tests and three
decision-emission/claim interruption tests passed. Final required
`bash scripts/agent_validate_changed.sh` passed 113 mapped tests, Ruff, mypy on
nine changed paths, compilation, forbidden-pattern checks, canonical live health
and the non-sending incident snapshot. Logs are private `/tmp/amzn-exit-*.log`.
The cost-model reader separately rejected an operational receipt with
`cost_metrics_missing_or_invalid`. No strategy/replay behavior changed; research
experiments and holdout evaluations were not run. Exact-tip CI remains required.

**Status:** repair locally validated; awaiting commit CI and after-close release
checks. October 2's historical decision/TCA gap remains unavailable evidence.
Acceptance after deployment: the next naturally occurring EOD reduction has one
causal operational decision, one durable claim, matching account/order/trace,
and a receipt resolving observed fill facts without certifying unknown fees.
Do not force a position or trade for this check.

## 2. October 2 accounting review

Fresh read-only broker check at **13:37 Pacific**: closed market, ACTIVE paper
account, no positions or active orders, complete pagination and six FILL
activities. Three one-share round trips: AAPL -$1.88, AMZN +$0.82 and MSFT -$0.02;
**-$1.08 gross** total. Broker equity change is also -$1.08. That equality does
not certify net fees, complete broker cash flows or individual fill identity.
All six fee amounts are unknown. Absence of fee activity does not establish zero.

The scheduled daily report was generated at **13:38:20 Pacific** for October 2.
All five local sources were stable during its read; no input-source gaps were
reported. Opening/closing position arithmetic matches all six fills. A separate
today-only comparison against the fresh complete broker capture passed all six
order quantities and execution counts. The comparison explicitly labels matching
counts `count_matched_identity_unverified`; it does not certify individual fill
identity. Private proof: `/tmp/ai-trading-accounting-close-20261002.json`.

The session report has five TCA matches and one missing AMZN EOD receipt. Its six
decision matches include an operational-order-role exception for that EOD exit;
there are only five ordinary decision rows in this narrower session audit. That
exception is not an original durable AMZN exit decision. The OMS funnel retains
the missing trace/account-chain gaps. All six fills lack a verified fee source
and total. Report status remains `evidence_pending`, with session reconciliation,
execution comparison and net-cost validation incomplete. The one historical
replay/execution comparison is insufficient and has an invalid net-fee contract.

The report's broader broker accounting window has 272 matching order quantities,
253 matching execution counts and 19 historical count mismatches. Today's six
orders all match in the bounded check; this does not resolve the older interval.
No account-level fees were allocated to guessed fills. Original records remain
unchanged. Report: `/tmp/paper-evidence-review-20261002.json`.

**Status:** today's report review and six-order quantity/count investigation
completed. Underlying historical AMZN lineage/TCA and verified-fee gaps are
blocked by unavailable evidence. Acceptance for net reporting requires genuine
broker-backed per-fill USD total fees, distinct from estimated/unknown costs;
historical identity gaps remain excluded from claims of complete causal evidence.

## 3. Paper operation and remaining actions

The service remains on tested release `7e2b81612` with zero automatic restarts.
Canonical broker state is fresh/flat and OMS consistency passes. The existing
`required_model_stale` and `replay_live_parity_gate_failed` flags remain. A passing
health/incident smoke check does not establish model qualification or live safety.

After exact-tip CI passes: recheck closed broker exposure, stage reviewed source
and release specification, pass both identity phases, restart after close and
verify broker/OMS health plus a non-sending incident snapshot. Keep rollback
source/spec available. Runtime EOD proof requires a future ordinary session.
Verified fees, legacy historical accounting and broader live-readiness evidence
remain separate blockers. Research budgets, holdout, model and trading gates
remain unchanged; no training, new experiment, forced trade or paid data was used.
