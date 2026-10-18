# Research reset amendment: October 18, 2026

**Decision:** the owner authorizes **one** registered development trial,
`day_sleeve_horizon_trial_20261018`, under the existing
`registered_development_trial` allowance of `config/research_reset.json`.
The pause otherwise continues unchanged. Next explicit review remains
**December 9, 2026**.

## Scope of the authorization

- The trial's contract is frozen in `config/day_sleeve_horizon_trial_campaign.json`
  and registered before any outcome is viewed, in
  `artifacts/day_sleeve_horizon_trial_20261018/campaign_state.json`.
- It reads only the verified 2024–2025 SIP development data for AAPL, AMZN and
  MSFT (`artifacts/stock_development/acquisition.json`). The September 9–December 8
  holdout is not opened; no 2026 data is read.
- It tests exactly one hypothesis: that the existing day-sleeve feature contract
  has positive out-of-sample edge after a fixed 10 bps round-trip cost when the
  label horizon is 60 minutes (the runtime hold contract) instead of five minutes.
  Rationale and evidence: [proposal](DAY_SLEEVE_HORIZON_TRIAL_PROPOSAL_20261018.md).
- Budget: one trial. A failing or inconclusive result retires the hypothesis and
  authorizes no tuning of threshold, horizon, features or model family.

## What a passing result permits

Planning only: a governed retrain of the day-sleeve model at
`AI_TRADING_DAY_SLEEVE_HORIZON_BARS=12` through the existing trainer and all
existing promotion, staleness and replay gates. It does not itself retrain,
promote, relax any gate, or change paper or live trading authority.

## What remains unchanged

- Scheduled training, new features, new model families and unregistered
  parameter searches stay paused (`training_block_reason()` still blocks the
  scheduled trainer).
- The 14-day staleness limit, promotion gates, replay/live parity gate and the
  `no_automatic_substitution_for_invalid_or_stale_selected_model` policy.
- The consumed ETF and September 21 replacement campaign budgets.
- The three named evidence gaps of the October 14 review (stale decision quotes,
  incomplete causal chains, per-fill fees) remain the reset's priority work.
- This amendment does not resume broad research and does not move the review date.

## Preconditions before the trial is claimed

1. `ai_trading.tools.model_replacement_trial` reads exit horizon, flatten cutoff,
   classifier specification and bootstrap seed from the contract (currently
   hardcoded to the five-minute trial); regression tests cover 60-minute exits,
   the 15:55 ET cutoff and non-overlap.
2. `ai_trading.tools.research_feasibility` provides a timestamp-only support audit
   for the intraday grid, and it passes for this contract.
3. Non-claiming preflight (source hashes, date bounds, feature provenance,
   runtime-equivalent history parity) passes.

Only then may `--fit` claim the trial, once, from a closed and flat window.

## Change record

- `config/research_reset.json`: added `registered_trials` and `amendments`;
  no other field changed. `resume_policy` stays `explicit_review_required`.
- `config/day_sleeve_horizon_trial_campaign.json`: status `registered`.
- `artifacts/day_sleeve_horizon_trial_20261018/campaign_state.json`: created by
  `ai_trading.tools.experiment_ledger.register_campaign` at
  2026-10-18T03:11:57Z with contract hash
  `46604293c025e39726e662b4060ab06f8cfefe7d4e6e2517eaf2c596802dce29`
  (immutable; zero trials claimed).
- Supersession note: a first registration at 03:10:04Z with hash
  `22e216d40e98de2d59a56eaf0cd1c82720d3bda83f2077ca3e2af9d51ccf5072` was
  replaced two minutes later, before any preflight, data read or claim, to add
  the structured keys the trial runner requires (`exit_horizon_minutes`,
  `flatten_cutoff_et`, `flatten_lead_minutes`, `classifier` with
  `early_stopping: false`). The hypothesis, data, horizon, cost, validation and
  acceptance are unchanged; only their representation moved from prose to keys.
  The superseded ledger is kept as
  `campaign_state.superseded_22e216d4.json`. No trial was claimed under either.
- Rollback: revert the two config edits and this record. Keep the ledger as
  evidence; a registered campaign is never silently unregistered.
