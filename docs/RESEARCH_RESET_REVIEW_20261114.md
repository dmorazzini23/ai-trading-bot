# Research reset review: November 14, 2026 (held early)

**Decision:** the pause on new work is ended. Portfolio manager mode may be
built, and new strategy research may start under pre-registered contracts.
Scheduled training stays blocked, the reserved holdout (September 9–December 8,
2026) stays unscored, consumed budgets stay consumed, and the real-money gate is
unchanged.

The owner held this review on November 14 instead of waiting for December 9.
**This deviates from the pre-registered
[December 9 review plan](RESEARCH_RESET_REVIEW_20261209_PLAN.md) and supersedes
its branch rules B and C.** That plan allows amendments only before the outcome
they concern has been viewed. This decision was made after all three trial
outcomes were known, so it is an owner override of the plan, not an application
of it. No holdout data was opened to make it.

## Evidence considered

All three registered trials finished in October. None showed an edge.

| Trial | Finished | Ledger status |
| --- | --- | --- |
| `day_sleeve_horizon_trial_20261018` ([record](RESEARCH_RESET_AMENDMENT_20261018.md)) | October 18 | `inconclusive_budget_consumed` |
| `cross_sectional_momentum_trial_20261018` ([record](RESEARCH_RESET_AMENDMENT_20261018_MOMENTUM.md#result-october-25-2026-retired)) | October 25 | `retired` |
| `volatility_target_core_trial_20261025` ([record](RESEARCH_RESET_AMENDMENT_20261025_VOLATILITY_TARGET.md#result-october-25-2026-retired)) | October 25 | `retired` |

With branch A of the plan closed on October 25, the December 9 review could
only have chosen B (retire the day-sleeve model) and C (continue the pause).
Waiting until December 9 would not have added a trial result. The other inputs
the plan names (weekly scorecards through December 5, the evidence-gap status,
and the parity gate and registry state on December 9) were not complete on
November 14 and were not used.

## What is now allowed

1. **Portfolio manager mode build**, per the
   [proposal](PORTFOLIO_MANAGER_MODE_PROPOSAL_20261025.md): offline first
   (its Phase 0), then paper (its Phase 1). The mode claims no edge.
2. **New strategy research**, under the rules already in force for registered
   trials:
   - one pre-registered contract per hypothesis, frozen before evaluation;
   - the timestamp-only sampling-support audit passes before any trial is
     claimed;
   - the owner's word is required for each claim;
   - a trial runs only if its campaign id is listed in `registered_trials` in
     `config/research_reset.json`, added by a dated amendment.

## What stays

- **Holdout.** September 9–December 8, 2026 stays unscored and reserved. No
  new contract may read it as development data.
- **Consumed budgets.** Every consumed campaign budget stays consumed. No
  re-runs and no variants of retired hypotheses on the same data.
- **Real-money gate.** The gate in the December 9 plan is unchanged. This
  review adopts no operational gate for portfolio manager mode, so that mode
  has no path to real money until a separate amendment sets one and the owner
  approves it.
- **Scheduled training.** Scheduled and automatic training stay blocked.
  `config/research_reset.json` keeps `enabled: true`, which is the only thing
  `training_block_reason()` in `ai_trading/config/research_policy.py` checks.
  Setting it to `false` would restart after-hours training on the live
  service; do not do that to "finish" this change.
- **Day-sleeve model.** It is to be retired from runtime, not retrained, in a
  separate later change. It keeps running as shadow-only until then.
- **Research automation.** While the policy is enabled, the daily, weekly,
  monthly and weekend research timers keep running only the three reset steps
  (reconciliation, session evidence review, scorecard). New research is
  started by hand under its contract, not by the timers.
- **Unregistered searches** stay forbidden.

## Explicitly not authorized by this review

- Any evaluation of the reserved holdout.
- Retraining or replacing the day-sleeve model, or relaxing staleness,
  promotion, replay or parity gates.
- Any live order, live profile or real-money step for either path.
- Deploying portfolio manager mode to the live paper service without its own
  validated change.

## Open items this review does not decide

- The owner decisions in section 6 of the portfolio manager mode proposal
  (mix, account type, amount, rebalancing rule, approval mode).
- The two evidence-gap questions added to the December 9 plan on November 8
  (quote ageing rule; whether `described_fee_allocation` closes the fee gap).
- Whether a review is still held on December 9. `review_date` stays
  `2026-12-09` in the policy file; it releases nothing automatically.

## Change record

- `config/research_reset.json`: `last_review_date`, `last_review_decision` and
  `review_record` point at this review. `allowed_work` gains
  `portfolio_manager_mode_build` and `registered_strategy_research`.
  `paused_work` is now `unregistered_search`, `scheduled_training`,
  `day_sleeve_retraining`, `holdout_evaluation` (`new_models` and
  `discretionary_features` removed). `enabled`, `review_date`,
  `resume_policy`, the holdout dates, `registered_trials`, `amendments` and
  `promotion_authority` are unchanged.
- No code reads `allowed_work`, `paused_work`, `last_review_*` or
  `review_record`; they are a record for people and agents. Runtime behaviour
  does not change. A test now pins that the checked-in policy keeps training
  blocked and lists `scheduled_training` as paused.
- Amendment appended to the December 9 plan; pointer added to
  `docs/RESEARCH_RESET.md`; `AGENTS.md` section 8 updated.
- Rollback: revert the commit. The policy keeps blocking training either way.
