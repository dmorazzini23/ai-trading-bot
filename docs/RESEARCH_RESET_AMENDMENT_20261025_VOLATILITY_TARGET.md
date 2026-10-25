# Research reset amendment: October 25, 2026 (volatility-target core trial)

**Decision:** the owner approved the drafted contract on October 25 and
authorizes a registered development trial,
`volatility_target_core_trial_20261025`, under the existing
`registered_development_trial` allowance of `config/research_reset.json`. The
pause otherwise continues unchanged. Next explicit review remains
**December 9, 2026**.

The momentum trial of October 18 was claimed and `retired` on October 25
([result](RESEARCH_RESET_AMENDMENT_20261018_MOMENTUM.md#result-october-25-2026-retired));
its hypothesis stays retired and its budget consumed.

## Scope of the authorization

- Contract: `config/volatility_target_core_trial_campaign.json`, proposal:
  [VOLATILITY_TARGET_CORE_TRIAL_PROPOSAL_20261025.md](VOLATILITY_TARGET_CORE_TRIAL_PROPOSAL_20261025.md).
  One hypothesis: SPY weight `min(1, 0.15 / 21-session realized volatility)`,
  rest BIL, 5-session rebalance with a 0.10 no-trade band. It must beat
  buy-and-hold SPY on Sharpe (positive 95% bootstrap lower bound at 10 bps) and
  have a maximum drawdown at most 0.85× a same-average-exposure static mix. It
  must also have a positive Sharpe difference at 20 bps, turnover ≤ 4×/year,
  ≥ 400 periods and ≥ 8 years with ≥ 40 periods.
- Data: daily SIP bars, `adjustment=all`, 2016-01-04 to 2025-12-31, acquired
  once by `ai_trading.tools.volatility_target_trial --acquire` *after*
  registration, with a hash manifest. No 2026 data and no holdout data is read.
- Budget: one trial. A failing or inconclusive result retires the hypothesis
  and authorizes no tuning of target, lookback, band, rebalance interval, cap,
  assets or cost.

## Clarifications made at registration (before any price was read)

The approved rules are unchanged. Three keys were added so the runner can
enforce the draft as written:

- `acquisition_manifest`: where the hash manifest lives.
- `weight_accounting`: the weight is held at its period-start value; drift is
  not modelled, and the controls use the same convention.
- `runner`: the tool that runs the trial.

## What a pass would permit

Planning only, under the [December 9 review plan](RESEARCH_RESET_REVIEW_20261209_PLAN.md):
one untouched consistency check on the holdout, then a registered paper
sleeve that must clear the real-money gate. No promotion, runtime or live
authority (`promotion_authority: false`, `runtime_authority: false`).

## Preconditions before the trial is claimed

1. Registration (this record, the config entries and the ledger).
2. `--acquire` writes the manifest.
3. `--preflight` passes: timestamp-only support audit, source hashes,
   2016–2025 bounds, price validity, price-validated support.
4. `--fit` only on the owner's explicit instruction, once.

## Change record

- `config/research_reset.json`: campaign appended to `registered_trials`, this
  record appended to `amendments`. `resume_policy` stays
  `explicit_review_required`.
- `config/volatility_target_core_trial_campaign.json`: status
  `draft_not_registered` → `registered`; `authorization` cites this record;
  the three clarification keys above.
- Ledger: `artifacts/volatility_target_core_trial_20261025/campaign_state.json`
  (runtime checkout, gitignored), created by
  `experiment_ledger.register_campaign` once this commit is deployed.
- Rollback: revert the config edits and this record. Keep the ledger as
  evidence; a registered campaign is never silently unregistered.

## Result (October 25, 2026): retired

Claimed once on the owner's instruction at 2026-10-25T15:52:15Z (market
closed), after the acquisition, support audit (498 periods, 10 years) and
non-claiming preflight passed. Ledger status `retired`. The holdout was not
evaluated; no model was fitted; no orders were sent.

| Criterion (contract) | Result | Met |
| --- | --- | --- |
| Adequate support | 498 periods, 10 of 10 years | yes |
| Sharpe difference vs buy-and-hold at 10 bps, 95% lower bound > 0 | +0.068 (0.936 vs 0.868); CI [−0.138, +0.256] | **no** |
| Max drawdown ≤ 0.85 × same-exposure static mix | 0.705 (19.0% vs 26.9%) | yes |
| Sharpe difference at 20 bps > 0 | +0.058 | yes |
| Annual turnover ≤ 4 | 1.15 | yes |

Reported only, at 10 bps: CAGR 13.3% vs 15.7% buy-and-hold and 14.2% static
mix; annual volatility 12.0% vs 15.9%; max drawdown 19.0% vs 29.9%
buy-and-hold; average SPY weight 0.89. The strategy trailed buy-and-hold in
8 of 10 calendar years and led in 2022 (−13.6% vs −18.5%); 2017 was equal.

The contract requires all criteria, so the hypothesis is retired. The
drawdown result does not rescue it: the pre-registered question was whether
risk-adjusted return improved with confidence, and ten years could not
distinguish the gain from zero. Per the stop rule, no tuning of target,
lookback, band, rebalance interval, cap, assets or cost is authorized.
