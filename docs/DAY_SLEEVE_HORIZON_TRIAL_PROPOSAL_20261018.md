# Proposal: registered day-sleeve horizon trial (draft, October 18, 2026)

**Status:** adopted and registered on October 18, 2026 by the
[reset amendment](RESEARCH_RESET_AMENDMENT_20261018.md), then run the same day.
Outcome **`inconclusive_budget_consumed`**; the hypothesis is retired. See the
[result record](DAY_SLEEVE_HORIZON_TRIAL_RESULT_20261018.md). Nothing below
authorizes a rerun or tuning.

## Why this is needed

The runtime has been in `paper_diagnostics_only` since the day-sleeve model
`ml_edge-histgb-236ba0fe-20260717200350-4045d5c5` passed its 14-day staleness
limit on about August 1. Every decision since carries `model_authority: false`
and evidence partition `stale_model_paper_diagnostic`; no trade counts toward
promotion, the replay/live parity gate cannot accumulate qualified samples
(97 of 250 at -7.0 bps on October 17), and the capital ramp cannot advance.

Nothing in the current policy can end that state before December 9:

- The scheduled trainer is blocked by `training_block_reason()` while the reset
  is active (`ai_trading/config/research_policy.py`).
- Its last attempt (September 9, 306,279 rows) was `skipped` with
  `no_qualified_candidate`: the logistic candidate failed all five selection
  constraints (`profitable_folds`, `profitable_fold_ratio`, `expectancy`,
  `score_separation`, `regime_stability`).
- The [October 14 review](RESEARCH_RESET_REVIEW_20261014.md) lists
  "retraining or replacing the day-sleeve model" under *explicitly not
  authorized*, and requires a pre-registered contract whose sampling-support
  audit passes before any new campaign claims a trial.

So the reset's `allowed_work` entry `registered_development_trial` is the only
admissible path, and it needs an explicit amendment because the October 14
review closed it for this model.

## Why retraining the same objective would fail again

The problem is the label, not the model age. Every candidate the project has
evaluated labeled a five-minute bar-open round trip net of a 10 to 12 bps cost:

| Evidence | Result |
| --- | ---: |
| Retired replacement trial, always-long control at 10 bps ([Sept 21](MODEL_REPLACEMENT_20260921.md)) | -9.95 bps net per opportunity |
| Retired replacement trial, fixed logistic, gross per selected trade | +0.22 bps |
| July 17 candidate manifest, mean realized training-label edge (cost floor 12 bps) | -11.93 bps |
| July 17 candidate manifest, positive-label rate | 18.2 % |
| 30-minute momentum study, gross break-even cost ([Sept 8](MOMENTUM_30M_RESULTS.md)) | 1.95 bps |

A five-minute gross move in AAPL, AMZN or MSFT is on the order of 1 bps; the
fixed cost is 10. The training target is structurally unwinnable, so any retrain
on that label reproduces `no_qualified_candidate` and consumes nothing but time.

Meanwhile the runtime itself does not hold for five minutes. It enforces a
600-second minimum and 7,200-second maximum hold and flattens 300 seconds
before the close. The model is being trained and judged on a horizon the
execution layer never trades. That mismatch is the one design variable no
registered trial has tested: the AAPL continuation study's 60-minute horizon
closed *inconclusive* only because validation support was 135 trades against
the 250 required, on one symbol.

## The trial

One frozen hypothesis, one trial, same conventions as the retired September 21
replacement trial except for the horizon:

- **Data:** the verified 2024–2025 SIP minute dataset for AAPL, AMZN, MSFT
  (`artifacts/stock_development/acquisition.json`, quality passed). No 2026 data
  is read; the September 9–December 8 holdout stays closed.
- **Features:** the existing day-sleeve feature contract, unchanged.
- **Decision / entry:** five-minute bar start plus 5 min 2 s; entry at the next
  one-minute open. Identical to the retired trial.
- **Exit:** one-minute open 60 minutes after entry, same session; opportunities
  whose exit would fall at or after 15:55 ET are excluded (the runtime flatten
  lead). No overlapping positions per symbol.
- **Label and cost:** net return minus a fixed 10 bps round trip; 6 and 20 bps
  reported as sensitivities, never as a rescue.
- **Model:** one fixed `HistGradientBoostingClassifier` specification, declared
  in the contract before any data is read. No family or parameter search.
- **Validation:** five chronological purged folds, training-only scaling,
  one-session embargo, whole-session bootstrap with seed 20261018.
- **Controls:** cash and always-long on identical opportunities.
- **Acceptance (same bar as September 21):** positive aggregate net edge per
  common opportunity at 10 bps with abstentions counted as zero, positive
  bootstrap 95 % lower bound, at least four of five profitable folds, at least
  250 selections per fold, improvement over always-long.

A pass authorizes **planning only**: a governed retrain of the day-sleeve model
with `AI_TRADING_DAY_SLEEVE_HORIZON_BARS=12`, through the existing trainer and
all existing promotion gates. It grants no promotion, runtime or live-money
authority. A fail or inconclusive outcome retires the 60-minute hypothesis and
does not authorize tuning.

## What has to be true before the trial can be claimed

1. **Owner amendment to the reset.** A new review record (draft text below)
   adopted by the owner, and `config/research_reset.json` updated to cite the
   campaign id. The code's `allowed_work` list already contains
   `registered_development_trial`; the amendment is a governance record, not a
   code change.
2. **Runner parameterization.** `ai_trading.tools.model_replacement_trial`
   hardcodes the five-minute exit (`labeled_rows`, lines 63–65, and the
   `exit horizon mismatch` check at line 108). It must read the exit horizon,
   flatten cutoff, classifier specification and bootstrap seed from the contract,
   with regression tests for 60-minute exits, the 15:55 cutoff and non-overlap.
   This is a correctness change to a research tool and is in scope under the
   reset.
3. **Intraday sampling-support audit.** `ai_trading.tools.research_feasibility`
   only audits the 60-session / 5-session ETF grid and raises
   `unsupported sampling contract` for anything else. The October 14 review
   requires a timestamp-only support audit to pass before a claim, so the tool
   needs a fixed-intraday-grid variant that counts possible opportunities per
   fold from timestamps alone, without reading prices.
4. **Registration before outcomes.** Register the contract with
   `ai_trading.tools.experiment_ledger.register_campaign` into
   `artifacts/day_sleeve_horizon_trial_20261018/campaign_state.json`. The
   contract hash is then immutable; `--fit` claims the single trial atomically.

Items 2 and 3 are the implementation work; estimate one focused session each.
Neither reads holdout data or fits a model.

## Command sequence (after the amendment and implementation land)

```bash
./venv/bin/python -m ai_trading.tools.research_feasibility \
  --protocol config/day_sleeve_horizon_trial_campaign.json \
  --sampling-acquisition artifacts/stock_development/acquisition.json \
  --output artifacts/day_sleeve_horizon_trial_20261018/sampling_support.json
```

```bash
./venv/bin/python -m ai_trading.tools.model_replacement_trial \
  --repository-root /home/aiuser/ai-trading-bot \
  --contract config/day_sleeve_horizon_trial_campaign.json
```

```bash
./venv/bin/python -m ai_trading.tools.model_replacement_trial \
  --repository-root /home/aiuser/ai-trading-bot \
  --contract config/day_sleeve_horizon_trial_campaign.json --fit
```

The first two commands read timestamps and build hashed partitions without
claiming anything. Only the third claims the trial. Run all of them from a
closed, flat window; none touches the trading service.

## Draft amendment text for the owner

> **Research reset amendment, {date}.** The owner authorizes one registered
> development trial, `day_sleeve_horizon_trial_20261018`, under the existing
> `registered_development_trial` allowance. Its contract is frozen in
> `config/day_sleeve_horizon_trial_campaign.json` and registered before any
> outcome is viewed. The trial reads only the verified 2024–2025 SIP development
> data for AAPL, AMZN and MSFT. The September 9–December 8 holdout, scheduled
> training, new features, unregistered searches and all promotion, staleness
> and replay gates remain unchanged. A passing result permits planning a
> governed retrain at a 60-minute label horizon through the existing trainer
> and gates; it does not itself retrain, promote or change trading authority.
> A failing or inconclusive result retires the hypothesis. This amendment does
> not resume broad research and does not move the December 9 review.

Change to `config/research_reset.json` when adopted: add
`"registered_trials": ["day_sleeve_horizon_trial_20261018"]` and a
`review_record` pointing at the adopted amendment file. No other field changes.

## What this proposal does not do

- It does not relax the 14-day staleness limit, the promotion gates, the replay
  parity gate or the `no_automatic_substitution_for_invalid_or_stale_selected_model`
  fallback policy.
- It does not open the holdout or use any 2026 data.
- It does not register, claim or run anything. All three are owner actions.
- It does not address the three named evidence gaps (stale decision quotes,
  incomplete causal chains, per-fill fees); those remain the reset's priority
  work and are unaffected.

## Rollback

Delete `config/day_sleeve_horizon_trial_campaign.json` and this document. If the
campaign was registered, keep `artifacts/day_sleeve_horizon_trial_20261018/`
as evidence; a consumed trial is never reopened.
