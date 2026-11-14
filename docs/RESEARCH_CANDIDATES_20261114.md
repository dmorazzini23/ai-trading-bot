# Research candidates: November 14, 2026

**Status:** proposal for the owner to adopt or reject. Nothing here is
registered. Nothing was run, fitted, claimed or scored to write it, and no
price file was opened. The only computation was a count of sessions from the
exchange calendar. `config/research_reset.json` is unchanged.

The top candidate has a complete draft contract:
`config/turn_of_month_exposure_trial_20261114_campaign.json`, status
`draft_not_registered`. The other two are outlined below.

## 1. What the four consumed campaigns say

| Campaign | Data | Outcome | Numbers from the report |
| --- | --- | --- | --- |
| Day-sleeve model, 5-minute label (`model_replacement_20260921`) and 60-minute label (`day_sleeve_horizon_trial_20261018`) | AAPL, AMZN, MSFT one-minute bars, 2024–2025 | not qualified; inconclusive | 60-minute: −0.85 bps net per opportunity, lower bound −1.49, 845 trades, 4 of 5 folds negative. Always-long on the same opportunities: −9.57 bps. |
| `research_reset_trend_20260908` | six ETFs, 2024–2025 | inconclusive, budget consumed | 4 common decision periods against 60 required; 84 opportunities excluded as not common to all six symbols. −5.4 bps per opportunity against +14.7 for always-long, on 24 opportunities in one quarter. |
| `cross_sectional_momentum_trial_20261018` | six ETFs, 2024–2025 | retired on all four criteria | 87 periods. −21.4 bps per period net (interval −76.7 to +31.1). −37.0 bps against always-long (interval −70.1 to −5.2). Always-long: +15.6 bps. 3 of 7 quarters profitable. |
| `volatility_target_core_trial_20261025` | SPY and BIL daily bars, 2016–2025 | retired on one criterion | Sharpe 0.936 against 0.868 for buy-and-hold: difference +0.068, interval −0.138 to +0.256. Drawdown 19.0% against 26.9% for the same-exposure mix. Turnover 1.15 per year. CAGR 13.3% against 15.7%. |

What the failures have in common:

- **Intraday horizons cannot pay the cost.** The model added 8.7 bps over
  always-long and still lost, because a 60-minute move is smaller than a
  10 bps round trip. The earlier five-minute proxy trades earned +0.22 bps
  gross. No change of features or model repairs that.
- **The trend trial tested nothing.** It failed on sampling support, not on
  the rule. The support audit that is now mandatory exists because of it.
- **Momentum lost to its own universe.** Picking one of six correlated ETFs
  each week was worse than holding all six, with the whole interval below
  zero. Two years and one selection per period is also a thin sample.
- **Volatility targeting had the right sign and too little data.** The
  drawdown, cost and turnover criteria passed. Ten years could not separate
  a Sharpe gain of 0.07 from zero.
- **Always-long won every comparison.** Holding SPY is the bar, as the
  December 9 plan already says.

The volatility result sets a limit on what any new test can show. That
strategy held 0.89 of SPY on average, so it moved almost exactly with the
benchmark, and its Sharpe-difference interval was still 0.39 wide. A rule
that moves less like SPY has a wider interval. With ten years of daily data,
no slow rule can show a higher Sharpe than SPY with 95% confidence unless the
true gain is about 0.3 or more. Contracts that demand that will be retired
whatever the truth is. The candidates below put the confidence requirement on
a paired, exposure-matched comparison, which ten years can resolve, and say
plainly that this is weaker than beating SPY.

## 2. Development data on disk

All paths are under `artifacts/` in the live checkout. Every dataset ends on
2025-12-31; none contains holdout rows.

| Dataset | Symbols | Bars | Range | Adjustment | Manifest |
| --- | --- | --- | --- | --- | --- |
| `research_foundation/expanded_data_repaired/40c7845c…` | DIA, IWM, QQQ, SPY, XLE, XLF | one-minute, SIP | 2024-01-01 to 2025-12-31, 502 sessions | split | `research_foundation/acquisition_sip_repaired.json` |
| `endpoint_proposal/adjusted_data/a6d219dd…` | same six ETFs | one-minute, SIP | same | all | `dataset.provenance.json` in the directory |
| `research_foundation/expanded_data/5b347584…` | same six ETFs | one-minute, IEX | same | split | `dataset.provenance.json` in the directory |
| `stock_development/data/eade0ac3…` | AAPL, AMZN, MSFT | one-minute, SIP | same | split | `stock_development/acquisition.json` |
| `volatility_target_core_trial_20261025/data` | SPY, BIL | daily, SIP | 2016-01-04 to 2025-12-31, 2,514 rows each | all | `data/acquisition.json`, with sha256 per file |

The only history longer than two years is SPY and BIL. There are no daily
bars for any bond, gold or sector ETF, and no event calendars.

**What the existing runners can score without new code: none of the
candidates.**

- `ai_trading/tools/volatility_target_trial.py` computes one weight rule,
  `min(1, target / realized volatility)`. It also requires the acquisition
  manifest to carry the contract's own hash, so it cannot reuse another
  campaign's files.
- `ai_trading/tools/registered_strategy_trial.py` scores a registry strategy
  that exposes `lookback`, `top_quantile` and `min_universe` on the six-ETF
  minute data. In practice that is cross-sectional momentum only.
- `ai_trading/tools/slower_horizon_study.py` is bound to the consumed trend
  contract and its ledger.

The first two refuse a contract whose status is not `registered`, or one
that `config/research_reset.json` does not list; the third reads only its own
fixed path. Each candidate needs a small runner, written and tested before
registration.

## 3. Three candidates, ranked

### Rank 1: turn-of-month exposure (SPY and BIL)

**Rule.** Hold SPY from the open of the last session of each month to the
open of the fourth session of the next month. Hold BIL otherwise. One
round trip per month; in SPY for 19% of sessions.

**Rationale.** US index returns have been concentrated around the turn of the
month since 1897, and in most developed markets (Ariel 1987; Lakonishok and
Smidt 1988; Ogden 1990; McConnell and Xu 2008). The usual explanation is
month-end cash flows: salaries, retirement contributions and coupon payments
that are invested within a few sessions.

**Why it is distinct.** The schedule depends on the calendar only. It uses no
price direction, ranking, volatility estimate or model.

**Data needed.** None new. It reuses the SPY and BIL daily files, verified by
hash. No Alpaca key is needed at any step.

**Why it might fail.**

- The effect has been public for forty years and may have decayed.
- It trades 24 times a year. At 10 bps per switch that is 20 bps a month, or
  2.4% a year. This is far more turnover than the volatility-target rule.
- The bar is high. SPY earned roughly 110 bps a month over BIL in this decade
  (15.7% CAGR in the volatility report). To pass, about 73 bps of that must
  fall inside the four-session window; a proportional share would be 21 bps.
  The older literature reports shares that large, but not for this decade.
- 119 monthly cycles give a standard error near 17 bps on the paired
  difference. A true net gain below about 33 bps a month will not pass.

**Relation to holding SPY.** It will almost certainly earn less than SPY in
total return, because it sits in bills four fifths of the time during a bull
market. It can only match SPY on Sharpe. Its use is for capital that would
otherwise be idle, which is the bot's current state. The contract requires a
Sharpe at least equal to SPY as a point estimate and reports the interval; it
does not claim to beat SPY with confidence.

**Fit with the portfolio manager.** The rule is a dated schedule of two
target weights. It could run inside the no-model mode only if that mode
accepts scheduled weight changes; that code is not on this branch and was not
checked.

### Rank 2: fixed diversified mix with tolerance-band rebalancing

**Rule.** Hold 60% SPY and 40% IEF. Rebalance to target only when either
weight is more than 5 percentage points from target at a session close,
trading at the next open. This is the rule the portfolio manager proposal
describes, and it settles that proposal's open question 4 (band width and
calendar frequency) for one mix.

**Rationale.** Treasuries have had low or negative correlation with equities
in most sell-offs, so a fixed mix lowers drawdown. Bands trade less than a
calendar rule for the same drift control (Masters 2003; Daryanani 2008).

**Why it is distinct.** Fixed weights. No signal of any kind.

**Draft acceptance, to be fixed in its own contract.** Maximum drawdown at
most 0.75 times buy-and-hold SPY; Sharpe at least equal to SPY as a point
estimate at 10 bps; turnover at most 0.5 per year. Reported only: the same
mix under quarterly calendar rebalancing, for turnover and drift.

**Data needed.** IEF daily SIP bars, `adjustment=all`, 2016-01-04 to
2025-12-31, acquired once into a hashed manifest. SPY would be re-acquired or
reused by hash. The owner may prefer AGG or BND; that choice must be made
before registration, not after.

**Why it might fail.** Bonds fell with equities in 2022, the largest
drawdown in the window, so the drawdown gain may be small. The decade was
strong for equities and weak for bonds, so the Sharpe criterion is more
likely to fail than pass. A two-asset band rule may trigger fewer than ten
times in ten years, which says little about the rule itself.

**Relation to holding SPY.** It will trail SPY in rising markets; the
portfolio manager proposal says the same. It is a risk choice, not an edge.
A retired result would not argue against running the portfolio manager,
which claims no edge. It would argue against expecting the mix to match SPY.

**Fit with the portfolio manager.** Direct. This is the candidate where the
two paths meet: the contract's weights and band are the mode's configuration.

### Rank 3: pre-FOMC drift (SPY and BIL)

**Rule.** Hold SPY from the close before each scheduled FOMC announcement to
the close on the announcement day. Hold BIL otherwise. Eight round trips a
year; about 80 events in 2016–2025.

**Rationale.** Lucca and Moench (2015) found that most of the US equity
premium from 1994 to 2011 was earned in the 24 hours before scheduled
announcements, about 49 bps per event.

**Data needed.** The list of scheduled FOMC announcement dates for
2016–2025 from the Federal Reserve's public calendar, frozen and hashed in
the contract. Prices are on disk. It needs close-to-close markouts, which
differ from the open-to-open convention of the earlier contracts, and
market-on-close orders, which the paper runtime has not been shown to place.

**Why it might fail.** Later studies report that the drift shrank or
vanished after publication, and this window starts after it. Eighty events
give a standard error near 12 bps, so only a net effect above about 25 bps
can pass, and each event costs 10 bps.

**Relation to holding SPY.** It is in the market 3% of the time. It is an
overlay for idle cash and cannot stand in for SPY.

### Why this order

1. **Turn of month** is the only candidate that needs no data and no keys,
   has the most independent periods (119), and rests on evidence that held
   across a century and many markets. Its weakness is cost.
2. **The fixed mix** is the most useful if adopted, because the portfolio
   manager can run it as written, and it has the lowest turnover. It ranks
   second as research because it is the least likely to match SPY on this
   decade, and the answer would not change whether the portfolio manager is
   built.
3. **Pre-FOMC drift** has the weakest recent evidence and needs both an
   event list and a new execution path.

Considered and left out: overnight-only SPY exposure (about 250 round trips a
year); weekly sector reversal (the sign-flip of the retired momentum rule on
the same data); post-earnings drift on three stocks (about 24 events);
inverse-volatility weighting across assets (a variant of the retired
volatility target); anything with a moving-average or lookback price filter
(a variant of the consumed trend rule).

## 4. Running the top candidate

Each step needs the owner's word. Steps 1 to 3 are prerequisites; 4 to 6 are
the trial.

1. Write `ai_trading/tools/calendar_exposure_trial.py` with regression
   tests. It does not exist. The commands below are its proposed interface,
   modelled on the volatility runner.
2. Adopt an amendment (`docs/RESEARCH_RESET_AMENDMENT_<date>_TURN_OF_MONTH.md`)
   and add the campaign id to `registered_trials` in
   `config/research_reset.json`.
3. Set the contract's `status` to `registered`, remove `draft_notice`, and
   register the ledger with `experiment_ledger.register_campaign`.

```bash
# 4. Support audit: session dates only, never claims
./venv/bin/python -m ai_trading.tools.calendar_exposure_trial \
  --repository-root . \
  --contract config/turn_of_month_exposure_trial_20261114_campaign.json \
  --support-audit

# 5. Preflight: source hashes, date bounds, price validity; never claims
./venv/bin/python -m ai_trading.tools.calendar_exposure_trial \
  --repository-root . \
  --contract config/turn_of_month_exposure_trial_20261114_campaign.json \
  --preflight

# 6. The single claim, from a closed market window
./venv/bin/python -m ai_trading.tools.calendar_exposure_trial \
  --repository-root . \
  --contract config/turn_of_month_exposure_trial_20261114_campaign.json \
  --fit
```

A result of `retired` or `inconclusive_budget_consumed` ends the hypothesis.
The window, execution time, assets and costs are not tuned afterwards.

## 5. Holdout

None of the three candidates needs September 9–December 8, 2026 during
development; every development window ends on 2025-12-31. The holdout holds
three month-ends and, by the published 2026 schedule, two FOMC announcements, too few to score either
calendar rule, so those contracts ask for forward paper observation after a
pass instead of a holdout score. The fixed mix could use the holdout once as
a consistency check under the December 9 plan.

## 6. Change record

- Added this document and the draft contract. No runtime code, test,
  `config/research_reset.json` or reset document changed.
- No test or tool enumerates `config/*_campaign.json`; each names its
  contract path. The draft is inert because both contract-driven
  runners refuse a status other than `registered`.
- Rollback: revert the commit.
