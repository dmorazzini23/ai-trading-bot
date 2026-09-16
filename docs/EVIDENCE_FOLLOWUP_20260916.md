# Four-item evidence follow-up — September 16, 2026

## September 15 reconciliation

All ten fills match decisions, orders and TCA records. All ten order quantities
match, and the complete opening/closing broker position snapshots reconcile.
Unknown account/mode count is zero. The residual session gap is fees_missing=10;
the aggregate fill_lineage_or_fee_gaps label did not mean ten broken joins.

A fresh, fully paginated two-day broker activity capture at 04:33 UTC contains
14 fills, 14 matched quantities, no fee fields and no fee activities. Therefore
no fee recovery or historical mutation is justified. Evidence:
/tmp/evidence-four-accounting.json and its private snapshot. Missing fees remain
unknown; no account-level charges were distributed across executions.

Fixed a real inconsistency: session reconciliation previously accepted broker
fee values without requiring USD/per_fill_total, and could derive fees from bps.
It now enforces the same total-fee contract as execution comparisons. The latter
also no longer exposes observed_fee_bps when the observed fee contract is invalid.
Session reports expose separate evidence_gap_counts so fee gaps are distinguishable
from unmatched lineage. These checks strengthen evidence requirements.

## Execution comparison diagnosis

The September 15 referenced replay contains 90 unique markout order IDs. Its two
overlapping actual trades are September 11 AMZN and September 14 AAPL:

| Trade | Observed benchmark | Simulation reference |
| --- | ---: | ---: |
| AMZN sell, Sep 11 | 255.20 | 255.17000000000002 |
| AAPL sell, Sep 14 | 333.02 | 332.94 |

These are real price differences, not floating-point equality noise. Replay
reference prices come from its limit/price fields; their equality to an actual
arrival benchmark cannot be assumed. No price was overwritten to manufacture a
pair. All ten September 15 fills lack matching markout observations in that
referenced artifact. A missing markout is not proof that an order never simulated.
The report now provides bounded per-order exclusion details and explicitly labels
its comparison scope as all supplied history, not just the selected session.
The thousands of account-identity exclusions are historical rows, not ten new
account capture failures. No replay search or holdout experiment was run.

## Natural production verification

Broker check at 04:34 UTC: market closed, zero positions/open orders and **zero
orders since the September 15 18:40:05 reporting deployment**. Thus there are no
natural production receipts yet to certify the new strategy/quantity fields.
Next regular open is September 16 at 13:30 UTC. Do not force an order. On the next
natural submission, join client/broker IDs across the broker, decision receipt
and durable intent; assert strategy agrees with the primary sleeve and requested,
submitted and filled quantities retain their separate meanings. Zero orders is
pending verification, never a passing result. Private broker snapshot:
/tmp/evidence-four-natural-orders.json.

## Training

The concrete review checklist is docs/TRAINING_RESUMPTION_REQUIREMENTS.md.
No training, model promotion, budget change or holdout evaluation was authorized
by this evidence work. Explicit review remains required to resume paused training.

## Scope and rollback

Validation: agent_validate_changed.sh --market-hours --skip-runtime-smoke passed
455 selected tests, lint, mypy (39 files) and compilation. Focused evidence and
scheduled-verification tests: 34 passed. The updated CLI was run against actual
September 15 records and retains evidence_pending with fees_missing=10 and zero
accepted comparison pairs. Non-sending incident snapshot passed. Broker health
was refreshed during the read-only accounting capture. Logs:
/tmp/evidence-four-{validation,focused,review}.log. Updated report:
/tmp/evidence-four-review.json. No strategy/decision behavior changed, so no
new backtest or research trial was needed.

Changes are in tools/execution_evidence_reconciliation.py and
tools/paper_evidence_review.py, with regression tests. Existing missing evidence
remains blocked. The next scheduled process imports these reporting changes;
no trading-service restart is needed for CLI-only reporting changes. Rollback
only these hunks if needed; no DB migration or historical data repair occurred.
