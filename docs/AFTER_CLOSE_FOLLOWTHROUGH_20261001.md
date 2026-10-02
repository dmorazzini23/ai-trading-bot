# October 1 after-close accounting and broker-read follow-through

Scope: the three authorized items; checked October 1 evening Pacific
(October 2 UTC). Runtime is deployed at `7e2b81612`. No strategy, model, research
budget, holdout, SDK, or trading gate changed. No orders or notifications sent.
Original execution/accounting records remain unchanged.

## 1. Authoritative broker-read latency

**Root cause:** each unrestricted active-order lookup also scans the complete
broker order history. The read-only baseline traversed **79 all-order pages,
39,427 response rows including overlap, in 4.027 seconds** after close.
The final candidate lookup returned zero active orders and positions, using
one open-order page plus the same 79 history pages in **4.081 seconds**.
These are after-close measurements, not a demonstrated improvement over the
previous session's slow scans.

A prototype using a submission-time discovery cursor was rejected: an order
newly visible to the client can have an older/missing submission timestamp and
be absent from the open filter. Point reads of previously known unresolved
IDs cannot discover that order. All-history reads remain authoritative.
The provider documents submission-time filters, not a complete mutation log:
[Alpaca order query](https://docs.alpaca.markets/us/reference/getallorders-1).

**Changes:** open queries now paginate as well as all-history queries; both
stream pages without retaining terminal history. Full pages with any missing
timestamp/ID, oversized pages, and nonadvancing cursors fail closed. Slow-scan
diagnostics include the open-page count. Freshness semantics remain unchanged.

**Regression coverage:** more than 500 open orders, overlapping/tied page
boundaries, malformed full pages, hidden pending orders newly appearing with
old/null timestamps, and reactivation/fill of old orders. The broader read
still runs when the open query is nonempty.

**Status:** pagination repair implemented; latency investigation complete;
latency reduction **blocked by unavailable completeness evidence** for a faster
discovery path. The patch does not claim faster scans.

**Remaining action and acceptance:** establish a provider-backed discovery
contract or design a separately reviewed durable event reconciliation path.
It must detect newly visible old/null-timestamp orders, inactive-order
reactivation, lost events, reconnects and account changes; fail closed on gaps;
match independent complete snapshots; and demonstrate lower request count and
latency under comparable conditions. Do not truncate history or introduce a
temporal cache to bypass this requirement.

## 2. Verified account attribution

**Root causes:** execution capture knew the broker account, but netting did
not carry it into submission metadata. Decision and execution/TCA allowlists
also omitted it. The canonical durable lifecycle writer independently rebuilt
metadata and dropped propagated account identity. Separately, a failed account refresh could retain a previous
`_evidence_account_id` for subsequent execution evidence.

**Changes:** decisions, including early rejections, use the current cycle's
broker account snapshot. The same ID and `broker_get_account` provenance enter
submission/intent metadata and execution/TCA records. Conflicting supplied
identity stops before submission. Account refresh/synchronization clears stale
evidence identity when the broker account is unavailable or lacks an ID.
The canonical lifecycle writer verifies the current account snapshot again
before claiming the intent and persists its ID/provenance. It uses the existing
per-cycle account cache; it never copies an unverified supplied ID into the store.
No database migration or historical backfill is performed. Unknown account
identity remains unknown; an identity match does not certify fill completeness.

**Regression coverage:** object/dictionary broker snapshots, unavailable
identity, rejected decisions, conflicting identity, account refresh failure,
account change, decision/OMS persistence and TCA propagation. A simulated
SQLite flow uses the actual lifecycle writer and OrderManager/IntentStore to
carry the account through durable decision, intent, submit claim,
acknowledgement and fill, and passes the funnel's account/causal linkage checks.
Separate writer tests cover matching/conflicting/unavailable broker identity.
These simulated acknowledgements/fills are local test fixtures, not broker
execution evidence or cost validation.

**Status:** repair tested in full CI, deployed and **verified in the observed
October 2 ordinary paper session**. The diagnostic found three intents/acks
(one canceled MSFT buy, two filled AAPL orders), three causal decision/intent
trace links and no gaps. All intent account IDs agree with the fresh broker
capture. Four pending/resolved TCA rows cover both filled broker order IDs and
carry that same account. The new release's ordinary run manifest was written
at 06:30 Pacific. Local intent/TCA identity checks do not establish complete
broker executions, verified fees or live performance. October 1's historical local account gap is
not repaired by this prospective change.

**October 2 closing follow-up:** later AMZN sell evidence reopens this item
for that exit path. The closing funnel has six successful trace links and one
`missing_trace_id`, on the AMZN sell. Its intent account matches the observed
broker, but the diagnostic reports `account_identity_consistent=false` and
`account_identity_unverified`. Root cause is not yet traced; no claim that the
broker used a different account is supported. The actual EOD path bypassed the
ordinary decision and TCA writers and discarded its genuine trigger time at the
durable writer. The prospective repair and validation are recorded in
`EOD_EXIT_ACCOUNTING_REPAIR_20261002.md`; CI/deployment remain pending. Historical
records and unknowns remain unchanged.

**Remaining action and acceptance:** continue ordinary monitoring for identity
conflicts and unknowns. The observed session now satisfies the prospective
account-path criterion; absent-account/failure behavior remains covered by
regressions. Historical gaps and verified fee evidence remain separate.
Never force a trade to obtain evidence.

## 3. October 1 execution fees

**Evidence limitation:** fresh raw `/account/activities` capture after October 1
00:00 Eastern completed pagination at **19:59:03 Pacific October 1
(02:59:03 UTC October 2)**. It returned exactly two MSFT fills and no
FEE/CFEE/PTC/PTR activities. Neither raw fill contains a fee amount. Raw GET
responses for both corresponding broker orders contain no fee, commission or
cost fields either. Consequently SDK model projection is not the cause for
these two orders. The existing capture uses the raw activity response and
preserves provider fields.

The raw order IDs, account, quantities and prices agree with the two local fill
observations. Entry time agrees exactly; local exit time differs by one
microsecond from the raw broker time. Both original timestamps are preserved;
this check does not manufacture a native execution-ID join. Existing session
evidence reports position arithmetic matched and an inferred FIFO observed
pair of **-$0.320 gross**, with **net P&L unavailable**. This is not broker lot
certification or a profitability claim.

The published
[Alpaca activity schema](https://docs.alpaca.markets/us/docs/account-activities)
does not promise a verified total execution fee on each FILL; nontrade
activities have their own accounting/settlement dates and net amounts. The
observed API evidence therefore cannot certify a complete total fee for these
fills. It does not establish zero fees or rule out later postings/statements.
Previously captured account charges remain unallocated to guessed executions.

**Changes/coverage:** no ingestion change is warranted by the observed raw
responses. Added a raw two-fill/no-fee regression: capture preserves the rows,
accounting retains two unknown total costs, and net-fee validation stays
unvalidated. Existing account-charge/nonallocation tests also pass.

**Status:** investigation completed; verified per-fill total fees **blocked by
unavailable provider evidence**, not resolved.

**Remaining action and acceptance:** revisit the same native order/activity
identities after broker postings or obtain a provider statement/confirmation
with exact account/order/execution linkage and a complete fee scope. Only then
can total fees, including an explicitly verified zero, support net reporting.
Keep verified fees, modeled costs and unknown costs separate. No purchase or
allocation of account-level charges is authorized.

## Validation and operations

- Required changed-file validation passed: 142 mapped tests, Ruff, mypy on
  13 paths, compilation, forbidden-pattern checks, live health and non-sending
  incident snapshot. Log: `/tmp/three-items-final-agent-validation.log`.
- Final streaming pagination adjustment additionally passed 29 affected tests,
  Ruff, mypy on two paths and compilation. TCA/account coverage is included in
  the mapped validation; its separate 11-test check also passed.
- Previous full validation for `d20388f88` remains historical evidence; the
  new release has its own exact-tip result below.
- Pre-deployment review caught the additional durable-writer omission before
  releasing `3da931cff`. That commit subsequently passed full exact-tip CI:
  7,327 tests passed, five skipped, 80.21% coverage, with all required workflows
  passing. Full log: `/tmp/ai-trading-ci-3da931cff.log`.
  Its follow-up passed 345 targeted tests, including runtime controls and the
  actual OMS writer/store path; standard validation passed 51 mapped tests,
  Ruff, mypy (three paths), compilation, forbidden-pattern checks, live health
  and non-sending incident snapshot. Log:
  `/tmp/three-items-durable-writer-validation.log`.
  Follow-up `7e2b81612d2745bdf1b3bcc7508e912ab36a0cd6` was approved/published;
  exact-tip CI `36961650029` passed 7,330 tests, five skipped, 80.21% coverage.
  CodeQL, SBOM, Workflow Lint, replay, research backtest and three determinism
  seeds passed. Full log: `/tmp/ai-trading-ci-7e2b81612.log`.
- Fresh pre-deployment broker check at 21:13 Pacific: closed, account ACTIVE,
  zero positions/active orders. Complete scan: one OPEN and 79 ALL pages,
  39,427 response rows including overlap, 3.858 seconds. This single after-close
  reading does not demonstrate a latency improvement under comparable load.
- Runtime checkout/spec staged at the follow-up; both installed identity
  phases passed. Restart completed at **21:15 Pacific October 1** (04:15 UTC
  October 2). Both automatic startup checks passed. After warm-up, health
  passed at 21:16 Pacific. Service remains active with zero automatic restarts;
  broker/OMS health
  is fresh and consistent. Structured health 503 retains only the existing
  `required_model_stale` and `replay_live_parity_gate_failed` qualification
  blockers. Non-sending incident snapshot passed; no test notification sent.
  It still recommends alerting for `edge_realism_gap_high`, `go_no_go_failed`,
  `go_no_go_failed_checks` and `health_degraded`; passing this smoke check
  validates snapshot construction, not trading or research qualification.
  Startup logs showed no structured warning/error entries in the inspected
  window. The latest run manifest still describes the previous session:
  its writer runs in the trading-cycle prelude, which has not run after close.
  New manifest and prospective account-path verification remain for the next
  ordinary session; no cycle or order was forced to manufacture evidence.
  Identity reports: `/tmp/ai-trading-followup-installed-{pre,final}.json` and
  automatic reports under `/var/lib/ai-trading-bot/runtime/`.
  Health: `/tmp/ai-trading-three-items-post-deployment.json`.
  Rollback spec: `/tmp/ai-trading-release-spec-before-3da931cff.json`.
- Diagnostic artifacts, including raw responses, remain host-local under
  `/tmp/ai-trading-fee-review-20261001*` and
  `/tmp/ai-trading-final-inventory-check*`; raw account identifiers are not
  published in this report.

Risk/rollback: account fields are additive and existing schema is unchanged.
Incomplete pagination now raises instead of returning a truncated snapshot.
Revert this patch and restore the previous reviewed release specification for
rollback, with exposure and release checks before any service restart. The
underlying live-capital goal remains blocked by its named evidence gaps.
