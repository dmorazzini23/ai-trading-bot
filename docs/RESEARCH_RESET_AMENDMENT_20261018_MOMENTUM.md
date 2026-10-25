# Research reset amendment: October 18, 2026 (cross-sectional momentum trial)

**Decision:** the owner authorizes a **second** registered development trial,
`cross_sectional_momentum_trial_20261018`, under the existing
`registered_development_trial` allowance of `config/research_reset.json`, and
adopts the `open_and_close_minutes` session-completeness rule exactly as
written in the contract. The pause otherwise continues unchanged. Next explicit
review remains **December 9, 2026**.

The first registered trial of this date (`day_sleeve_horizon_trial_20261018`,
[record](RESEARCH_RESET_AMENDMENT_20261018.md)) was claimed, run and closed
`inconclusive_budget_consumed`; its hypothesis is retired and its budget stays
consumed.

## Scope of the authorization

- The trial's contract is frozen in
  `config/cross_sectional_momentum_trial_campaign.json` and registered before
  any outcome is viewed, in
  `artifacts/cross_sectional_momentum_trial_20261018/campaign_state.json`.
- It reads only the verified 2024–2025 SIP development data for the six-ETF
  universe DIA, IWM, QQQ, SPY, XLE, XLF
  (`artifacts/research_foundation/acquisition_sip_repaired.json`). The
  September 9–December 8 holdout is not opened; no 2026 data is read.
- It tests exactly one hypothesis: that the unmodified
  `CrossSectionalMomentumStrategy` (63-session lookback, top quantile 0.2,
  min_universe 6), scored on buy signals only with non-overlapping 5-session
  holds, earns positive net returns after a fixed 10 bps round trip and beats
  an equal-weight always-long basket on identical dates. Rationale and
  evidence: [proposal](MULTI_DAY_STRATEGY_TRIAL_PROPOSAL_20261018.md).
- Session-completeness rule (owner decision): a session counts if its first
  and last regular minutes are present and every present minute has valid
  price geometry and positive volume; no minute-coverage floor, no
  interpolation, no invented bars. DIA's 973 missing minutes across 243
  sessions include no opening or closing minute (largest single-session gap 16
  minutes). The 98 percent coverage alternative was audited first from
  timestamps only and blocked; it is disclosed in the proposal.
- Budget: one trial. A failing or inconclusive result retires the hypothesis
  and authorizes no tuning of lookback, quantile, hold, universe, cost or
  completeness rule.

## What a passing result permits

Planning only. It establishes development-period evidence for one existing
registry strategy. It does not retrain or promote any model, relax any gate,
enable the strategy in the runtime, or change paper or live trading authority
(`promotion_authority: false`, `runtime_authority: false`).

## What remains unchanged

- Scheduled training, new features, new model families and unregistered
  parameter searches stay paused.
- The 14-day staleness limit, promotion gates, replay/live parity gate and the
  `no_automatic_substitution_for_invalid_or_stale_selected_model` policy.
- The consumed ETF trend, September 21 replacement and day-sleeve horizon
  campaign budgets.
- The three named evidence gaps of the October 14 review remain the reset's
  priority work.
- This amendment does not resume broad research and does not move the review
  date.

## Preconditions before the trial is claimed

1. Timestamp-only daily support audit
   (`ai_trading.tools.research_feasibility`) passes for this contract.
2. Non-claiming preflight (`ai_trading.tools.registered_strategy_trial
   --preflight`) passes: source hashes, 2024–2025 date bounds, no 2026
   timestamp, complete-session panel.

Only then may `--fit` claim the trial, once, on the owner's explicit
instruction. Registration alone (this record) claims nothing.

## Change record

- `config/research_reset.json`: `cross_sectional_momentum_trial_20261018`
  appended to `registered_trials`; this record appended to `amendments`; no
  other field changed. `resume_policy` stays `explicit_review_required`.
- `config/cross_sectional_momentum_trial_campaign.json`: status
  `draft_not_registered` → `registered`; `authorization` cites this record.
  No other key changed.
- `artifacts/cross_sectional_momentum_trial_20261018/campaign_state.json`:
  created by `ai_trading.tools.experiment_ledger.register_campaign` at
  2026-10-18T23:56:26Z with contract hash
  `5f694a82e5b07900ad2767d6947c29b33795b49c1c98c2e0298c6033e66e0a3e`
  (immutable; zero trials claimed). Neither preflight nor support audit has
  been run under this registration.
- Rollback: revert the two config edits and this record. Keep the ledger as
  evidence; a registered campaign is never silently unregistered.

## Result (October 25, 2026): retired

Claimed once on the owner's instruction at 2026-10-25T04:03:29Z (closed
window), after the support audit (87 periods, 7 supported quarters) and the
non-claiming preflight passed. Ledger status `retired`; report sha256
`64827b8e…30fa`, periods sha256 `5e9c7c70…b8b8`, signature `c9611baa…`.
The holdout was not evaluated; no model was fitted; no orders were sent.

| Criterion (contract) | Result | Met |
| --- | --- | --- |
| Adequate support | 87 periods, 7 quarters | yes |
| Net vs cash at 10 bps, 95% lower bound > 0 | mean −21.4 bps/period, CI [−76.7, +31.1] | no |
| Paired vs equal-weight always-long, lower bound > 0 | mean −37.0 bps/period, CI [−70.1, −5.2] | no |
| Aggregate net at 20 bps > 0 | −31.4 bps/period | no |
| At least 5 of 7 profitable quarters | 3 of 7 (2024Q4, 2025Q2, 2025Q3) | no |

Equal-weight always-long earned +15.6 bps/period net at 10 bps on the same
dates. The hypothesis is retired; per the stop rule, no lookback, quantile,
hold, universe, cost or completeness tuning is authorized.
