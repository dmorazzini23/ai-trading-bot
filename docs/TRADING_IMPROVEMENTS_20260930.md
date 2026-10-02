# September 30 follow-through: five improvements

Evidence captured October 1 UTC, after the September 30 Pacific session.
This is correctness and existing-evidence work under the research reset. No
training, search, experiment, holdout evaluation, forced order, ticker expansion
or gate change was performed. Implementation completion and evidence completion
are different statuses.

## 1. Remove the redundant KPI broker scan

**Root cause:** `_emit_cycle_execution_kpis` fetched the entire active-order
snapshot again solely to report pending orders. The verified September 30
broker measurement required 80 calls / 39,425 history rows / 4.33 seconds for
one complete scan. That does not quantify its share of market-session latency.

**Change:** reporting reuses the synchronized inventory only when its order
component is fresh, the read is known and its monotonic age is 0–60 seconds.
Missing, failed, future or older snapshots produce null pending-count/age
metrics with explicit availability and source fields. All authoritative broker
queries, complete pagination and submission/exposure gates remain unchanged.
Known pending orders with unknown creation time are counted; their unknown ages
are reported separately and the oldest age remains null rather than implying zero.

**Regression:** valid inventory produces counts and aged-order diagnostics
without any broker call; five unavailable-inventory cases remain unknown and
also make no broker call. Existing KPI/SLO regressions remain applicable.

**Status:** implemented, deployed and verified in the first October 1 session
snapshots. Reporting reused synchronized inventory ages 0.547/0.690 seconds,
with `open_orders_snapshot_ms=0`; pending count changed from one to zero after
the fill. Authoritative broker and OMS state were fresh/consistent after warm-up.
Whole-cycle latency is not resolved: required complete scans still read 79 pages
and 39,425 orders, taking about 14–35 seconds during startup. Acceptance for a
future bounded optimization: maintain complete authoritative inventory and
fail-closed unknowns, demonstrate lower complete-cycle time against a comparable
session baseline, and preserve exposure consistency. No profitability claim.

## 2. Explain where orders stop, without invented conversions

**Root cause:** summary counts mix decision records, submit claims and actual
broker activity. A submit claim is made before local go/no-go checks and does
not establish a network submission. Rejected outcome records also lost the
pre-submit decision timestamp and trace identity, defeating causal matching.

**Change:** the netting outcome recorder now retains decision/source timestamps
from the existing pre-submit intent context and carries its trace on rejection.
Original historical records are preserved. The new read-only `order_funnel`
tool uses one SQLite read transaction, durable decision UUIDs, explicit trace
identities, symbols/sides and causal times. Ambiguous/missing identities and
conflicting accounts remain gaps. Receipt/bar timestamps are never substituted.
The daily paper evidence review includes this diagnostic automatically using
managed OMS configuration; `DATABASE_URL` takes precedence. An explicit
`--oms-database` can identify an isolated diagnostic copy.

`OK_TRADE` means final accepted outcome. Pre-submit qualification is not separately
recorded in this durable history, so that stage's count is **null**, not guessed
from later outcomes. Reported gate reasons include nonblocking scales and can
overlap. A durable `SUBMIT_ACK` with the same broker identity is reported as local
acknowledgement, not independent broker proof. Local fee columns are not used to
infer verified costs.

Read-only September 30 result: **476 durable decision records, zero final
accepted outcomes; 12 intents and submit claims; zero durable broker acks or
recorded fills**. All twelve local intent/decision joins fail the causal check
on the historical outcome timestamp, and account identity is absent. They remain
unjoined. Their recorded final intent error is the runtime go/no-go rejection.

Operator invocation from a tested checkout:

```bash
./venv/bin/python -m ai_trading.tools.order_funnel \
  --database /var/lib/ai-trading-bot/runtime/oms_intents_paper_monday.db \
  --session 2026-09-30 --output /tmp/order-funnel-20260930.json
```

**Regression:** rejected outcome retains pre-submit time and trace; diagnostics
distinguish claims/acks/fills, reject duplicate/ambiguous identities, account and
causality conflicts, honor offset/day boundaries and URL precedence, never
create missing databases, and integrate with the daily review.

**Status:** diagnostic and prospective defect repair resolved. Historical causal
and account gaps are blocked by unavailable evidence. Acceptance for operational
closure: deployed daily report appears and newly recorded attempts preserve
causal identities; any account/provenance gaps remain explicit. There is no
authorization to repair history by backdating or guessed joins.

## 3. Strengthen fault evidence

**Evidence:** existing process-level fixtures already cover broker acceptance
before acknowledgement, death, acknowledgement storage failure, unresolved
exposure gates, partial fills, duplicate/reordered events, and cancel/fill races.
The previous release covers cache expiry immediately before submit claim.

**Change:** the durable simulator now records every submission invocation.
Assertions compare invocation count to distinct broker orders, so broker-side
deduplication cannot hide duplicate calls. New fixtures race two process owners
for the same durable intent and exercise a partial fill followed by cancellation
and duplicate delayed acknowledgements. They preserve one submit attempt and
the recorded fill quantity/terminal status.

**Status:** resolved within the isolated simulator scope. Eight process tests
passed; no fault was injected into the running service. These are local process
tests, not a real broker network-partition or cross-host ownership drill.
Acceptance for this patch: process suite, cache-expiry regression and release
checks pass. Real two-host fencing remains item 4's external evidence gap.

## 4. Preserve the live-risk boundary and prepare a real owner drill

**Established defect:** equity risk checked receipt age but could accept an old
source observation received just now. It now checks both source and receipt age
against the existing 60-second bound. The new regression rejects an observation
two minutes old with a current receipt. Fourteen equity-risk tests passed.
Approved numeric limits, cash adjustments, reserves, atomic high-water behavior
and fail-closed live opening behavior are unchanged.

**Evidence limitation:** the deployed credentials and ledger are paper; no
approved funded live starting/session/high-water baseline or complete timed live
cash-flow interval has been supplied. The provider's account creation timestamp
is not an equity observation timestamp. Date-only/delayed activities cannot
prove intraday absence of cash movements. These limitations are documented in
`SMALL_ACCOUNT_RISK_CONTRACT.md`. No opening balance or source time was invented.
The pinned SDK inspection confirms `TradeAccount` exposes `created_at`, `equity`
and `last_equity` but no equity observation time, while `NonTradeActivity` uses
`date`. This matches the [documented account models](https://alpaca.markets/sdks/python/api_reference/trading/models.html#alpaca.trading.models.TradeAccount)
and [activity schema](https://docs.alpaca.markets/docs/account-activities).
It is a limitation of the available evidence, not proof that an unexamined live
account can never supply any additional evidence.

The completed isolated off-host restore and integrity checks are reused.
SQLite paper operation does not establish cross-host submission ownership. The added opt-in real
PostgreSQL test acquires the actual advisory lock in one process, proves a
second cannot acquire it, kills the owner, then proves takeover after the backend
releases it. It creates no tables and is restricted to a disposable local
`ai_trading_owner_test` database distinct from the runtime database. With no
configured test server it explicitly skips. For this task, Ubuntu PostgreSQL
16.15 packages were downloaded and unpacked into `/tmp` without sudo or a system
installation. A temporary server bound only to `127.0.0.1:54329` ran the actual
two-process advisory-lock test: **one test passed in 5.24 seconds**. Its EXIT
cleanup stopped the server. No production database, schema, service, account or
broker connection was involved. Log: `/tmp/five-items-real-postgres-drill.log`.

For a future disposable server, keep credentials out of
chat and use the managed `AI_TRADING_TEST_OWNER_DATABASE_URL` setting, then run:

```bash
./venv/bin/pytest -q tests/integration/test_postgres_owner_recovery.py
```

An actual controlled noncurrent-version restore also passed on October 1 UTC.
The previously verified September 29 bundle was uploaded twice with identical
bytes but different phase metadata under a new key in the approved
`pruned/recovery_backups/` prefix. The first version, now noncurrent, was fetched
by its exact returned version ID; version/metadata checks and SHA-256 matched.
The original backup objects were untouched. The existing approved 30-day
retention applies to this fixture. The isolated restore recovered **3,045
entries**, with both SQLite integrity checks `ok`; file restoration took 11.813
seconds. At 04:44:58 UTC, fresh broker evidence confirmed the same paper account
and flat positions as the restored boundary, zero active orders, and a closed
market. The archive's code identity is the historical `edfefa72f` release;
no service was started from it and restored order authority remains disabled.
This is a point exposure comparison, not full cash/fee/history reconciliation.
Evidence: `/tmp/five-items-version-recovery-20261001T043836Z/proof.json` and
`broker-boundary-proof.json`. The enabled backup timer's September 30 daily run
also succeeded. Same-content version selection proves retrieval mechanics; it
does not supply missing historical versions or prove cross-host failover.

**Status:** source-freshness repair, real local owner-crash drill and controlled
noncurrent-version restore resolved;
live equity/cash and cross-host recovery proof remain blocked by unavailable
evidence/infrastructure. Acceptance requires the
same live account's approved baselines, complete timestamped cash reconciliation,
source-backed equity observations, atomic enforcement across opening paths,
the already-passing real PostgreSQL drill, and a separate cross-host stop/revoke/fence restore
rehearsal. Absent original historical versions remain unavailable. A passing
local owner test alone cannot satisfy all these requirements. No live activation.

## 5. Concrete research decision from completed results

**Decision: retire `fixed_logistic_day_model_after_costs_v1` as a replacement
candidate under its consumed one-trial campaign.** Collect ordinary operational
evidence to verify repairs, execution fidelity and cost observability; that is
not a justification to rerun the hypothesis. There is no new experiment in this
patch. Any genuinely different hypothesis requires its own frozen specification
and approval before evaluation.

The September 21 frozen development report and saved out-of-fold predictions
cover 2024–2025. Existing artifacts reproduce the following decomposition:

| Existing model evidence | Value |
| --- | ---: |
| Opportunity rows | 94,200 |
| Selected proxy round trips | 30,802 (32.70%) |
| Development sessions / sessions with selection | 416 / 412 |
| Selected entries per development session, all three symbols | 74.04 |
| Mean gross return per selected proxy trade | +0.221377 bps |
| Frozen round-trip cost assumption | 10.000000 bps |
| Mean net return per selected proxy trade | −9.778623 bps |
| Gross contribution per common opportunity | +0.072387 bps |
| Cost contribution per common opportunity | −3.269851 bps |
| Net return per common opportunity | −3.197464 bps |
| Always-long net / cash control per same opportunity | −9.947166 / 0 bps |
| Paired improvement over always-long | +6.749701 bps |
| Session-bootstrap 95% lower bound | −3.445699 bps |
| Profitable chronological folds | 0 of 5 |

The better result relative to a losing always-long control does not establish
positive economics. The selected average gross edge is far smaller than the
frozen cost. More samples cannot rescue this already well-sampled negative result.
Entry frequency is a turnover proxy; account-level notional turnover and executable
profit are not established by these bar-open returns. No cost assumption was tuned.

The existing September 30 replay reports **195 markout samples** (250 required),
**−7.294058 bps** candidate net edge and **−7.940112 bps** baseline edge; the
caps-only comparison improves it by **0.646054 bps**, still negative. It has
319 candidate simulated orders / 929 baseline orders and 197 / 524 fill events.
Its signed reference-to-fill cost averages are −4.107016 / −2.748726 bps; these
are simulated price/fee accounting metrics, not verified round-trip broker fees.
The headline is next-observation markout, not linked entry/exit realized P&L.
Do not combine this metric with the development trial's five-minute round-trip
returns or treat the sample shortfall as its only failure. Its existing loss
attribution explicitly cannot identify exits, queue position or impact.

**Evidence verification:** no trial or replay was rerun. Report/ledger/prediction
hashes match the existing ledger and report; only saved aggregate/OOF evidence
was read. The untouched holdout, consumed campaigns and all promotion gates
remain unchanged. No replacement serving artifact was created.

| Source | SHA-256 |
| --- | --- |
| `artifacts/model_replacement_20260921/report.json` | `0b8b61fcc6787edbc212eead64d6f133f4e04556900c04bcedb5fad75e064034` |
| `artifacts/model_replacement_20260921/campaign_state.json` | `f00fdc092e9c15285fb8ccf45e4f7606e0e9f889645b35701b13cf09f5722b85` |
| `artifacts/model_replacement_20260921/oof_predictions.parquet` | `c2649569ecfe3e08ac52133e33190db6e2292ec8933ae6cc707c49854657584e` |
| `/var/lib/ai-trading-bot/runtime/replay_outputs/replay_hash_20260930.json` | `00f74e4d18e060c66aa6161bf3baee15af4b4772488fba48217cecce39774c4a` |

**Status:** research decision resolved. Qualification is still blocked, and
profitability is unproven. Next action is operational verification of deployed
repairs; a future new study must specify a different economic mechanism,
predeclared development data, execution/cost contract, control, stopping budget
and acceptance rule, preserving the September 9–December 8 holdout. No such
study is authorized or run here.

## Validation and deployment boundary

Full host validation passed: **7,308 tests passed, two skipped**, Ruff, mypy,
strict type checks and tracked Python compilation. The PostgreSQL test skipped
in the general suite passed against the isolated real server separately; the
other skip is an existing regime fixture without a regime column.
Focused and standard validation results, runtime checks and
release status are recorded in `CODEX_HANDOFF.md`. Exact-tip CI passed with
**7,305 tests passed, five skipped and 80.20% coverage**. The owner restarted
the paper service at 07:30 Pacific October 1 on tested release `d20388f88`;
both automatic startup identity checks passed and the process has zero automatic
restarts. It was an open-market owner-run restart; no further restart was
performed. After warm-up, the broker and OMS checks were fresh/consistent,
with one MSFT paper share and zero active orders at 07:35 Pacific.
The daily funnel captured one local decision/intent/ack/fill chain using a causal
pre-submit decision timestamp; account attribution remains unavailable locally.
No old records were rewritten. The non-sending incident snapshot passed;
no notification was sent. HTTP 503 still reflects stale-model and replay
qualification, and the startup log contains the existing stale-model error.
No model, replay, freshness, fee, provenance or promotion threshold was relaxed.

Rollback: deploy the preceding tested code/specification through the existing
release-identity procedure after broker exposure review. No schema/configuration
migration or historical data rewrite is included. Reporting may show unknowns
where the old code implied zero; consumers must retain that distinction.
