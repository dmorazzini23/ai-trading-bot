# Research reset review: October 14, 2026

**Decision:** keep the pause. Scheduled training, new models, unregistered
searches and discretionary features stay paused through the reserved holdout
(September 9–December 8, 2026). Work is limited to named evidence gaps and
correctness fixes. Next explicit review: **December 9, 2026**.

This review was due October 8 and was recorded late, on October 14. Nothing was
resumed in between: `review_date` never releases work automatically
(`ai_trading/config/research_policy.py`).

## Evidence considered

| Source | Observation |
| --- | --- |
| Campaign ledger | Both registered trials are consumed: `liquidity_reversal_foundation_20260908` ended `inconclusive_budget_consumed`. The `research_reset_trend_20260908` trial budget is spent, and its sampling support allows at most 24 selections against the 100 required. |
| Retired replacement trial | Fixed logistic replacement: +0.22 bps gross, −9.78 bps net at the frozen 10 bps cost ([follow-through](TRADING_IMPROVEMENTS_20260930.md)). |
| Replay/live parity gate (health, Oct 14) | `replay_counterfactual` fails: candidate net edge −8.15 bps from 103 samples, against a required 0 bps and 250 samples. OMS lifecycle parity is clean: 2,236 intents, 0 violations. |
| Day-sleeve model | `ml_edge-histgb-236ba0fe…` was trained July 17 and is 89 days old against a 14-day limit. It runs only as shadow with `serving_decision=abstain`. The last training attempt (Sept 9) was skipped with `no_qualified_candidate`. |
| Weekly scorecard (Oct 13) | `evidence_pending`. 0 of 22 observed fills have complete causal chains. 1,896 of 2,399 quote rows were rejected as more than 1,000 ms old. Fee coverage: 168 fills `total_fee_unknown`. |
| Paper session (Oct 14) | Three one-share round trips (AAPL, MSFT, AMZN), ≈ −$0.68 gross. These are diagnostics only and not qualified-model performance. |

No evidence supports a qualified candidate, a rescued result, or a new campaign.
Repeating trials on consumed budgets or loosening gates would only fit noise.

## Named evidence gaps (in priority order)

1. **Stale decision quotes.** Most quote rows are more than 1 s old at record time.
   Trading cycles that run over budget (open-order snapshots that scan the full
   order history, ~17 s each) are a likely contributor. Commit `893f491`
   (symbol-scoped order reads, [repair notes](CYCLE_ACCOUNTING_REPAIR_20261012.md))
   targets this and is not yet deployed. Account-wide discovery remains a
   separate follow-up.
2. **Incomplete causal execution chains.** Fills lack a verified causal quote,
   and some lack an explicit decision time or account/order link. Repair the
   writers going forward; leave historical records unchanged.
3. **Per-fill total fees.** These remain unavailable from the paper Trading API
   activity schema. Do not impute zero fees or allocate account-level charges.

## Explicitly not authorized by this review

- Retraining or replacing the day-sleeve model, or relaxing staleness, promotion
  or replay gates.
- Any evaluation of the reserved holdout before December 9.
- New campaigns or variants of consumed hypotheses.

A new campaign after December 9 requires a pre-registered contract whose
timestamp-only sampling-support audit passes before a trial is claimed.

## Change record

- `config/research_reset.json`: `review_date` moved to `2026-12-09`, and
  `last_review_date`/`last_review_decision` recorded. `resume_policy` stays
  `explicit_review_required`.
- Rollback: revert the commit. The policy keeps blocking training either way.
