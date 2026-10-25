# Proposal: volatility-targeted market core (registered trial draft)

**Date:** 2026-10-25
**Status:** draft for the owner. Nothing is registered, claimed or fitted. No
price has been read or analysed; one timestamp-only availability request was
made (below). The holdout (2026-09-09 to 2026-12-08) and all 2026 data stay
closed.

Contract draft: `config/volatility_target_core_trial_campaign.json`
(`"status": "draft_not_registered"`).

## 1. Why change direction

Every registered trial that tried to pick winners has failed after costs:

| Campaign | Hypothesis | Outcome |
| --- | --- | --- |
| `liquidity_reversal_foundation_20260908` | ETF opening-shock reversal | inconclusive, budget consumed |
| `research_reset_trend_20260908` | 60-session ETF trend filter, 5-session hold | inconclusive, budget consumed (24 of 100 selections possible) |
| `day_model_replacement_20260921` | fixed logistic day model | retired (−9.78 bps net) |
| `day_sleeve_horizon_trial_20261018` | 60-minute day-sleeve horizon | inconclusive, budget consumed |
| `cross_sectional_momentum_trial_20261018` | top-momentum ETF, 5-session hold | retired (−21.4 bps/period; always-long +15.6) |

In each case the always-long control did better than the selection rule. This
proposal stops trying to beat the market by selection. It starts from holding
the market and tests one published risk rule that may improve the
*risk-adjusted* result: hold less when realized volatility is high.

## 2. Which ideas are still open

- **Volatility targeting: open.** No campaign used a volatility signal; the
  September trend contract explicitly excluded it.
- **Long-horizon price-trend filter (e.g. 10-month average): closed.** It is a
  price-direction in/out rule on the same kind of ETFs as
  `research_reset_trend_20260908`. Under the October 14 review, a variant of a
  consumed hypothesis is not authorized, so it is not proposed.
- **Disciplined fixed-mix rebalancing:** not a hypothesis. It claims no edge
  and needs no trial, only an owner decision about allocation if a core is
  ever run.

## 3. Hypothesis

`spy_bil_volatility_target_15pct_21_session_5_session_rebalance_v1`: every 5
sessions, set the SPY weight to `min(1, 0.15 / σ)`, where σ is the annualized
standard deviation of the last 21 daily log returns. Hold the rest in BIL
(T-bills). Skip changes smaller than 0.10. Trade at the next open. No leverage
or shorts.

It passes only if **all** of the following hold at a 10 bps round-trip cost on
traded weight:

1. The Sharpe ratio is higher than buy-and-hold SPY, with a positive 95%
   block-bootstrap lower bound.
2. The maximum drawdown is at most 85% of that of a **static** SPY/BIL mix with
   the same average SPY weight. This blocks the trivial win of "less exposure
   means smaller drawdown".
3. The Sharpe ratio is still higher than buy-and-hold at a 20 bps cost (point
   estimate).
4. Annual turnover is at most 4× the portfolio.

There must also be at least 400 periods and 8 calendar years with 40+ periods
each. Expected: about 498 periods over 10 years.

The bar is strict on purpose. The published effect on a single index is modest,
so a fail is a realistic outcome. A fail would also be useful: it would say the
plain market hold (or a fixed mix) is the honest default.

## 4. Data

- **Window:** 2016-01-04 to 2025-12-31, daily SIP bars, `adjustment=all`
  (dividends included).
- **Coverage:** the timestamp-only check on 2026-10-25 found SPY and BIL with
  2,514 bars each across that window. It covers the 2018 Q4, 2020 Q1 and 2022
  sell-offs.
- **Not untouched:** 2024–2025 ETF data was viewed in earlier research. No
  registered trial used 2016–2023 daily data; earlier unregistered model
  research may have, so it is treated as not untouched either.
- **Holdout:** not read. After a pass, the December 9 plan's one-time
  consistency check applies (about 12 periods; it cannot show significance).

## 5. Work needed before a claim (each step on the owner's word)

1. **Adopt.** The owner reviews this contract (parameters are frozen once
   registered), adopts a research-reset amendment, and registers the campaign
   ledger. This can happen on December 9 or earlier by amendment, as with the
   momentum trial.
2. **Tooling (offline only).**
   - The existing `registered_strategy_trial` runner scores buy signals and
     cannot score continuous weights, so a small dedicated runner is needed.
   - A daily acquisition step with a hash manifest is also needed.
   - Both must refuse any 2026 row before prices are read.
   - Tests and validation as usual; nothing touches the live bot.
3. **Checks.** A timestamp-only support audit, then a non-claiming preflight.
4. **Claim.** `--fit` once, then report against section 3.

## 6. What a pass would and would not mean

- **A pass permits planning only.** That means one holdout consistency check,
  then a registered paper sleeve. It must clear the December 9 plan's
  real-money gate: 8 weekly scorecards, positive after measured costs, at least
  even with exposure-matched SPY, `live_canary_allowed`, and owner approval.
- **It would not beat the market on raw return.** With exposure capped at
  100%, the expected benefit is a smoother ride with similar risk-adjusted
  return, not higher returns.
- **The bot would become a portfolio manager**, not a stock picker. It would
  rebalance on schedule and enforce the risk rule. The intraday selection logic
  would be retired.

## 7. Change record

- Added this proposal and the draft contract. `config/research_reset.json` is
  unchanged. No ledger was created and no data was acquired.
- Rollback: delete both files.
