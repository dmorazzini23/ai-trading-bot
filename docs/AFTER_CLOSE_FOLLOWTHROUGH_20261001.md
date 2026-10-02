# October 1 after-close accounting and broker-read follow-through

Scope: the three authorized items; checked October 1 evening Pacific
(October 2 UTC). Runtime remains at `d20388f88`. No strategy, model, research
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
also omitted it. Separately, a failed account refresh could retain a previous
`_evidence_account_id` for subsequent execution evidence.

**Changes:** decisions, including early rejections, use the current cycle's
broker account snapshot. The same ID and `broker_get_account` provenance enter
submission/intent metadata and execution/TCA records. Conflicting supplied
identity stops before submission. Account refresh/synchronization clears stale
evidence identity when the broker account is unavailable or lacks an ID.
No database migration or historical backfill is performed. Unknown account
identity remains unknown; an identity match does not certify fill completeness.

**Regression coverage:** object/dictionary broker snapshots, unavailable
identity, rejected decisions, conflicting identity, account refresh failure,
account change, decision/OMS persistence and TCA propagation. A simulated
SQLite flow carries the account through durable decision, intent, submit claim,
acknowledgement and fill, and passes the funnel's account/causal linkage checks.
These simulated acknowledgements/fills are local test fixtures, not broker
execution evidence or cost validation.

**Status:** repair implemented and locally verified; **awaiting CI/deployment
and ordinary session evidence**. October 1's historical local account gap is
not repaired by this prospective change.

**Remaining action and acceptance:** publish the reviewed commit with explicit
approval, pass its exact-tip CI and release identity checks, recheck closed/flat
broker exposure, then deploy after close. Inspect new ordinary paper decisions
and any resulting intents/TCA: when a broker account is available, identities
must agree with it and the account gap must disappear; unavailable identity
must remain visibly unknown. Never force a trade to obtain this evidence.

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
- Previous full validation for the deployed release remains evidence for
  `d20388f88`, not this new commit. New exact-tip CI is still required.
- Candidate read-only broker check: closed, zero positions/active orders.
  Running service stays active with zero automatic restarts; broker/OMS health
  is fresh and consistent. Structured health 503 retains only the existing
  `required_model_stale` and `replay_live_parity_gate_failed` qualification
  blockers. No restart or deployment has occurred in this task.
- Diagnostic artifacts, including raw responses, remain host-local under
  `/tmp/ai-trading-fee-review-20261001*` and
  `/tmp/ai-trading-final-inventory-check*`; raw account identifiers are not
  published in this report.

Risk/rollback: account fields are additive and existing schema is unchanged.
Incomplete pagination now raises instead of returning a truncated snapshot.
Revert this patch and restore the previous reviewed release specification for
rollback, with exposure and release checks before any service restart. The
underlying live-capital goal remains blocked by its named evidence gaps.
