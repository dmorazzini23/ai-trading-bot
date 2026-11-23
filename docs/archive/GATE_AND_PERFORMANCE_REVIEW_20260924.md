# September 24 paper gate and performance decision

This is an after-close evidence review, not a strategy trial or a profitability
claim. It uses the paper account ending `f0f6`; no order was sent for this review.

## 1. Runtime reconciliation block

The 19:56 UTC runtime go/no-go report failed `closed_trades`, `win_rate`,
`acceptance_rate`, `open_position_reconciliation_consistent`, and
`replay_live_parity_gate_consistent`. Its reconciliation input is the unbounded
local `trade_history.parquet`, not the broker-backed rebuilt ledger. It
reconstructed AAPL -3 and AMZN +1 while the broker was flat. These are the
same two differences documented before the verified flat September 18 opening;
the origin before the September 7 03:19:26 UTC anchor remains unresolved.
The reviewed September 7–18 and September 22 quantity intervals are separate.
No opening inventory or missing transaction has been invented.

The September 24 session audit observed flat broker positions at 13:29:33 and
20:00:47 UTC, accepted zero fills, and matched opening-to-closing quantity
arithmetic. The 20:37 UTC activity capture was pagination-complete from June 26
and had no September 23 or 24 activities. A later direct broker check found
zero positions and open orders. This supports no new September 24 position
discrepancy; it does not certify the legacy interval or execution-feed
completeness from quantity agreement alone.

The report's `current_basis` authority label was misleading for a reconstruction
from unbounded local events. It now labels reconstructed inventory
`diagnostic_only`, records `reconciliation_evidence_scope=unbounded_local_events`,
and states `verified_broker_ledger_applied=false`. The numerical mismatch and
every go/no-go failure remain visible and blocking. The existing flat-broker
regression now asserts these labels and the preserved discrepancy.

**Status:** investigation complete; historical evidence gap unresolved. A
current-position reconciliation can replace this diagnostic only after a
complete same-account broker position opening, a covering execution-complete
activity interval with identities, and a closing/current broker position are
verified as one scope. Preserve the original history and the pre-anchor
exclusion rule. Do not make the runtime gate pass by dropping legacy rows or
using the broker position as its own comparison ledger.

## 2. Performance and research decision

September 24 had no broker orders, broker fills, or accepted session fills.
The session audit's zero-fill position arithmetic matched, but its status is
`evidence_gaps`; no realized strategy P&L is established. The cost comparison
has one slippage pair against 30 required orders and zero paired net-cost
orders. Broker accounting still has no complete execution-linked total-fee
contract. An absent new fee row does not establish zero fees. The runtime
report's five-day diagnostic figures (19 closed trades, 31.6% win rate,
0.75% acceptance) are below the configured 50, 46%, and 1.5% thresholds;
its estimated P&L is not verified net strategy performance.

At the after-close health check, replay was fresh but failed the counterfactual:
186 candidate samples versus 250 required and -1.492 bps candidate net edge
versus a positive requirement. More samples alone would not establish positive
economics. `required_model_stale` remains the sole health readiness failure;
paper diagnostics have no promotion authority. The September 21 registered
replacement hypothesis remains retired after its one consumed, negative
development trial. The September 9–December 8 holdout remains untouched.

**Decision:** continue routine operational evidence collection and keep the
model, replay, cost, freshness, provenance, promotion, and runtime go/no-go
gates. No training, parameter search, strategy trial, ticker expansion, forced
trade, or paid data access was performed. A different research hypothesis
would need its own registered specification and approval. Verified net
performance needs complete causal execution chains, broker-supported total
costs, sufficient independent samples, positive net economics, and the
existing qualification process; September 24 supplies none of that by itself.

## Validation

The focused report/lineage suite passed 75 tests. Changed-file Ruff, mypy,
compile, and 792 mapped tests passed with bytecode directed to `/tmp`; the
default sandbox run had two audit-test filesystem failures because `.codex`
hook bytecode paths are read-only there. The validator's localhost smoke request
was also sandbox-denied; the host smoke returned the expected HTTP 503 with
`required_model_stale`, an active service, and zero restarts. A read-only run
against the production report inputs retained both AAPL/AMZN mismatches and
reported the new diagnostic-only labels. Exact-tip CI and deployment identity
checks are separate release gates.

Evidence: September 24 daily `paper_evidence_review.json` and
`broker_account_activities.json` under
`/var/lib/ai-trading-bot/runtime/research_reports/daily/20260924T203749Z_daily/`,
`runtime_performance_report_latest.json`, current broker/health reads, and
`docs/ACCOUNTING_AND_RESEARCH_DECISION_20260923.md` for the established
historical boundary and consumed trial.
