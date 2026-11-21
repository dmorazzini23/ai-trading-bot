# Research reset amendment: November 20, 2026 (weak-close trial)

**Decision:** the owner approved the specification as drafted on November 20
("Sounds good. Let's do it") and authorizes one registered development trial,
`weak_close_trial_20261120`, under the `registered_strategy_research` allowance of
the [November 14 review](RESEARCH_RESET_REVIEW_20261114.md). The claim and the
single scoring (`--fit`) need the owner's separate word. Scheduled training stays
blocked, the holdout stays unscored, consumed budgets stay consumed and the
real-money gate is unchanged.

- Contract: `config/weak_close_trial_campaign.json`, `status: registered`.
- Runner: `ai_trading/tools/weak_close_trial.py`.
- Ledger: `artifacts/weak_close_trial_20261120/campaign_state.json` (gitignored).

## Hypothesis

The owner picked it: a "weak close" in a broad index fund tends to be followed by
a rebound. For SPY and for QQQ, each on its own: when the price at 15:45 New York
time sits in the bottom fifth of the session's range so far, buy in that session's
closing auction and sell in the next session's closing auction, holding on without
a trade while the signal repeats. It must earn more than cash after 1 bp per side,
and more than holding each fund every session at the same average weight.

The measure is internal bar strength, IBS = (close − low) / (high − low). The
reference is A. S. Pagonidis, "The IBS Effect: Mean Reversion in Equity ETFs"
(2013); a replication is Pandey and Joshi (2023, arXiv 2306.12434). **The original
paper could not be found and remains unread.** The source precondition was met from
secondary sources only: they confirm the citation (2013; the first draft said
2014), the IBS below 0.2 long threshold and a next-session rise, with a reported
average of +0.35% after IBS below 0.20 and −0.13% after IBS above 0.80. A detail of
the original that the secondary sources omit (for example a different exit) cannot
be ruled out; it would not change the registered contract.

## The fixed specification

One specification; no grid, no tuning.

| Item | Fixed value | Why |
| --- | --- | --- |
| Instruments | SPY and QQQ, each judged separately | The owner's choice |
| Signal | IBS < 0.2 (strictly) at 15:45, from the 15-minute bars stamped 09:30 to 15:30: high and low of those bars, price = close of the 15:30 bar | The published bucket; the close itself cannot be known before the close |
| Zero range | No signal | IBS is undefined |
| Entry | Market-on-close order in the same session, filled at the daily close | The broker's closing-auction cut-off is 15:50; 15:45 leaves five minutes |
| Exit | Market-on-close the next session; if the fund signals again at 15:45 that day, no sale (one round trip per run of signals) | The published one-session hold, without selling and rebuying at the same price |
| No entry | Early closes (13:00), sessions missing a snapshot bar, the last development session | No reliable snapshot, or an exit in 2026 |
| Sizing | Half of equity per fund; both signal = 100% invested; one signals = the other half in cash; no leverage, no resizing while held | No sizing parameter to fit |
| Costs | 1 bp per side decides; 2 bps per side must still beat the control; 0.5 and 5 bps reported | Auction fills pay no spread against the auction price; fees on sales are under 0.3 bps |
| Benchmarks | Cash; and the **exposure-matched control**: each fund at the strategy's average weight in it, every session, no cost | A long-only rule in a rising market beats cash with no skill; against the control only timing counts |
| Statistic | Mean net return per close-to-close interval (all ~2,513 of them, zero when in cash), and the mean paired difference against the control | |
| Uncertainty | Circular block bootstrap, 21-session blocks, 10,000 replicates, seed 20261120; lower bound = 1.25th percentile | As in the last three contracts; see multiple testing |

**Pass (all required):** adequate support; lower bound of the mean net return
above zero at 1 bp; lower bound of the mean paired difference against the
exposure-matched control above zero at 1 bp; mean paired difference above zero at
2 bps; mean paired difference above zero in at least 6 of 10 calendar years; mean
paired difference above zero for SPY alone and for QQQ alone.

**Retire:** support adequate and any criterion failed. No change to the
threshold, snapshot time, exit, instruments, sizing, costs or acceptance follows.
The textbook version (final daily IBS, entry at that same close) is reported as
an upper bound and cannot rescue a fail.

**Inconclusive (budget still consumed):** support inadequate after the claim, for
example fewer than 300 signal sessions for either fund.

**Support thresholds:** 2,300 observations, 9 years with at least 230, 2,300
sessions per fund with a complete snapshot (checked from time stamps before the
claim), 300 signal sessions per fund (checked after the claim).

**A pass permits** one check on the reserved holdout (about 60 sessions, about 12
signal sessions per fund: a consistency check, not significance), then at least
eight weeks on the paper account. No promotion, runtime or real-money authority.

## How this differs from the retired swing-reversal trial

| | `swing_reversal_trial_20261115` (retired) | This trial |
| --- | --- | --- |
| Universe | 300 most liquid US common stocks; funds excluded by name | Two index ETFs, SPY and QQQ |
| Signal | Trailing 5-session return relative to the stock's sector, ranked across 300 names; buy the bottom 30 | Where today's 15:45 price sits in today's range (one session, intraday shape); no ranking; each fund on its own |
| Entry | Next session's **open** | Today's **closing auction** |
| Horizon | 5 sessions on a fixed grid, with a hold band | To the next close; about 1.25 to 1.4 sessions on average |
| Benchmark | SPY at the same weight on the same days | The same fund at the same average weight on every day |
| Data | Daily bars of ~760 stocks, 2016–2025 | Daily and 15-minute bars of SPY and QQQ, 2016–2025 |

Both belong to the "buy recent weakness" family and both use 2016–2025, and SPY
was the swing trial's control series. But the instruments do not overlap (the swing
universe excluded funds), the signal is a different quantity measured over a
different period, and the trade is entered at a different time and held for a
quarter of the time. This is not the swing-reversal hypothesis under a new name.

## Overlap with the other contracts

- `liquidity_reversal_foundation_20260908` (inconclusive, budget consumed) is the
  **closest relative**: it bought six ETFs including SPY and QQQ after a large fall
  in the first 30 minutes and sold two hours later, the same day, on 2024–2025
  data. Same reversal family, same two funds among its six, overlapping years. It
  differs in trigger (an opening shock against a range-scaled threshold, not the
  close's position in the range), timing (intraday only, no overnight hold) and
  window. The owner should confirm this is a separate hypothesis.
- `intraday_momentum_trial_20261115` (retired): SPY last half hour after a
  positive first half hour, 2016–2025. Different signal and an opposite
  (momentum) mechanism, but the same SPY bars, and a weak close is partly a weak
  last half hour.
- `research_reset_trend_20260908`, `cross_sectional_momentum_trial_20261018`:
  60- and 63-session trend rules on six ETFs including SPY and QQQ, 2024–2025.
- `volatility_target_core_trial_20261025`: SPY sized by trailing volatility,
  2016–2025.
- `day_model_replacement_20260921`, `day_sleeve_horizon_trial_20261018`: intraday
  models on three stocks. `opening_range_trial_20261118`: intraday breakout in
  single stocks.
- Unregistered drafts `turn_of_month_exposure_trial_20261114` and
  `fixed_mix_trial_20261115`: calendar and allocation rules.

None tests where the close sits in the day's range.

## Usable date range

**2016-01-04 to 2025-12-31.** The provider serves intraday SIP bars from
2016-01-04, and the shared development-date check refuses every 2026 row (the
holdout, September 9 to December 8, 2026, is never read). No part of that window is
untouched: SPY 2016–2025 was read by four trials and SPY and QQQ 2024–2025 by three
more. QQQ for 2016–2023 has not been read by any registered trial. Budgets are
spent per hypothesis, not per date range, so the window can be reused for a
distinct hypothesis; the reuse is counted under multiple testing.

## Data needed

From Alpaca, SIP feed, `adjustment=all`, 2016-01-04 to 2025-12-31, SPY and QQQ
only: daily bars (one request per fund) and 15-minute bars (one request per fund
per calendar year). About 22 requests, about 45 pages with extended-hours bars, well
under a minute of the shared rate limit. Kept: about 65,000 fifteen-minute bars and
2,514 daily bars per fund, about 5 MB. Acquired once, after registration, outside
market hours, with a hash manifest.

The acquisition refuses, writing nothing: any 2026 or out-of-window row; a regular-
hours bar off the 15-minute grid or on a non-session date; any missing or
duplicated daily bar; more than 1% of full sessions missing a snapshot bar. The
preflight blocks without a claim if a daily bar is invalid or if more than 1% of
sessions have a daily close more than 1% away from the last 15-minute close.

## Expected counts (from the definition, not from data)

About 2,492 sessions per fund can carry a signal. If IBS is roughly uniform, about
20% signal: about 500 per fund. With a 20–30% chance that a signal repeats the next
session, about 350–400 round trips per fund, 700–800 in all (70–80 a year), average
hold 1.25–1.4 sessions. The funds will often signal together; the account would hold
something on roughly a quarter of sessions, at an average exposure of about 0.2 of
equity.

**Power, stated before any data (assumptions, not measurements):** with about 500
signal sessions per fund, a next-session standard deviation near 1.3% and the two
funds strongly correlated, the lower bound clears zero only if a signal session
beats the average session by roughly 12 bps or more. Costs (about 0.3 bps of equity
per session) do not decide this; decay of the effect does. If it has halved since
publication, a fail is the likely outcome.

## Multiple testing

One hypothesis. By the opening-range contract's count, eleven hypotheses had been
tested on overlapping data and none passed; this is the twelfth. The lower bound
stays at the 1.25th percentile; the chance of at least one false pass in twelve
independent attempts at that level is roughly 14%.

## Known weaknesses

- **Widely published and reportedly weakened.** IBS and "buy a close near the low"
  rules are standard practitioner material. The test asks whether anything is
  left, net, in 2016–2025.
- **Two instruments that move together.** SPY and QQQ will often signal on the same
  day; the effective sample is closer to one fund than two.
- **Closing-auction execution through a retail API.** The order must reach the
  broker before its cut-off (assumed 15:50, unverified) and be routed to the auction; whole shares only.
  A late or rejected order means no trade or an after-hours fill, neither
  modelled. The paper account cannot confirm auction fills.
- **15:45 is not the close.** The last 15 minutes are among the busiest of the day;
  the 15:45 signal and the final IBS will disagree on some sessions (reported).
- **The daily close is taken to be the auction price.** The preflight check (1%)
  would not detect a systematic difference of a few basis points.
- **Not untouched data**, and a related reversal rule on the same funds was tested.
- Overnight and weekend gap risk is in the returns; no stop, by design. Taxes are
  ignored.

## Owner decisions, November 20, 2026

Accepted as drafted:

- entry in the closing auction on a 15:45 signal, not the next open (the overnight
  share of the return is reported);
- exit at the next close, holding through a repeated signal;
- two fixed half-equity slots;
- 1 bp per side decides, 2 bps required; 0.5 and 5 bps reported;
- the exposure-matched same-fund control decides, with a per-fund criterion;
- the 1.25th-percentile lower bound;
- the 2024–2025 opening-shock ETF trial (`liquidity_reversal_foundation_20260908`)
  is a separate hypothesis from this one;
- a missing daily bar refuses the acquisition, with no automatic tolerance; any
  gap goes back to the owner before scoring;
- no probe mode.

Settled or recorded:

- **Source:** met from secondary sources only; the original is unread (above).
- **Close price:** the daily close is taken to be the auction price, as drafted;
  it remains a known weakness.
- **Open before any paper run (does not block the historical test):** the broker's
  closing-auction cut-off (assumed 15:50) and its handling of fractional or
  whole-share market-on-close orders are UNVERIFIED. Its documentation does not
  list market-on-close among fractional order types, and a forum report describes
  partial fills of such orders on the paper account. Recorded in the contract under
  `open_items_before_paper`.

## Validation of the runner

Synthetic data only; no network or provider call. `tests/tools/test_weak_close_trial.py`
covers the signal (strict threshold, zero range, the 15:45 snapshot), no look-ahead
(changing everything from 15:45 on, including the day's final daily bar, moves no
decision), entry and exit at daily closes, costs (two sides per round trip,
nothing for a repeated signal), the exposure-matched control (drift alone fails),
the per-fund and stress and year criteria, inconclusive support, holdout and 2026
refusal at every stage, degraded downloads, the close cross-check, and a single
claim (a second fit, or any stage after the claim, returns
`budget_already_claimed`; a crash after the claim consumes the budget).

## Change record

- `config/research_reset.json`: campaign appended to `registered_trials`, this
  record appended to `amendments`.
- `config/weak_close_trial_campaign.json`: `draft_not_registered` → `registered`;
  `draft_notice` removed; `owner_decisions` and `open_items_before_paper` added;
  the source corrected to 2013 and its read status recorded; preconditions marked.
  Every rule, parameter, threshold, cost and the seed are unchanged from the draft.
- Ledger: `artifacts/weak_close_trial_20261120/campaign_state.json`, created with
  `experiment_ledger.register_campaign` before any data is acquired.
- Acquisition after the close on November 20, on the owner's word; support audit
  and preflight follow. No claim until the owner's separate word.
- The running bot's checkout and data directory are not touched.
- Rollback: revert the config edits and this record; keep the ledger as evidence.

## Registration, acquisition and checks, November 20, 2026 (nothing scored)

- **Registered** at 20:27 UTC with `experiment_ledger.register_campaign`. Contract
  hash `77e145c4950f55827704d9a02e080605ebc6f8b029b4928f9cd69e0b58f6c5b6`; the
  ledger holds no trial.
- **Acquisition** after the close, 21:01 to 21:06 UTC, on the owner's word: 22 bar
  requests (10 fifteen-minute and 1 daily per fund), paced 3 seconds apart.
  Credentials were hydrated in memory only. Manifest
  `artifacts/weak_close_trial_20261120/data/acquisition.json` (sha256
  `b34a742e...49301a0a`). Every file runs 2016-01-04 to 2025-12-31 with no 2026
  row. Daily bars: 2,514 per fund, every session. Fifteen-minute bars kept: SPY
  65,111, QQQ 65,110 of 65,112 expected (2,493 full sessions, 21 early closes).
- **Gaps (bar stamps only):** SPY 2019-08-12 lacks its 15:45 bar, which the signal
  does not use; QQQ 2016-02-22 lacks its 10:30 and 10:45 bars, so QQQ gives no
  signal that session (`missing_snapshot_bars`). Both are within the 1% limit;
  nothing was waived.
- **Support audit** (stamps only): passed. 2,513 observations, 250 to 253 a year,
  10 supported years; sessions with a complete snapshot: SPY 2,492, QQQ 2,491.
- **Preflight** (no claim): passed. No invalid bars. Signature
  `b260067991d8ad267afc173ff2879313711735cd22eb529b80e97fc63c7044a7`.
- **Close cross-check (a data check, not a result):** the daily close and the last
  regular fifteen-minute close differ by a median 0.8 bps and a 99th percentile of
  8 bps for SPY. Two SPY sessions differ by more than 1%: 2020-03-12 (+2.9%) and
  2020-03-18 (−2.2%), both market-wide circuit-breaker days in March 2020; QQQ has
  none. Two of 2,514 sessions is under the 1% limit, so the preflight passes. The
  test fills at the daily close, so on those two sessions the result depends on
  the provider's daily close being the official closing price. Recorded before
  any scoring; no rule changed.

No signal, return or strategy figure was computed before the claim.

## Result: retired (November 21, 2026)

The owner accepted the two March 2020 SPY closes as recorded and gave the word
("run the fit"). The single claim ran on 2026-11-21 04:10 UTC with evidence
signature `b260067991d8ad267afc173ff2879313711735cd22eb529b80e97fc63c7044a7`;
report SHA-256 `39837b55e4e22e21011adff7163cea28dcee14624158de87da4ed60e4776d6a5`.
The budget is consumed. The trial is **retired**: it met five of the six
conditions and missed one.

| Condition (deciding cost 1 bp per side unless stated) | Value | Met |
| --- | --- | --- |
| Adequate support | 2,513 sessions, 824 signals, 702 round trips | yes |
| Lower bound against cash | +0.87 bps per session | yes |
| Lower bound against the exposure-matched same-fund control | −0.13 bps per session | **no** |
| Mean difference against the control at 2 bps per side | +1.36 bps per session | yes |
| Positive years | 9 of 10 (2018 negative) | yes |
| Each fund positive against the control | SPY +0.77, QQQ +0.87 bps | yes |

Reported figures at the deciding cost: net 2.81 bps per session (about 7.1% a
year, Sharpe 0.79, maximum drawdown 11.9%) at an average exposure of 16%; mean
difference against the control 1.64 bps per session. The yearly difference
against the control is 7.9 bps per session in 2020 and 4.8 in 2025, between 0.1
and 2.4 in seven other years, and −4.2 in 2018; the advantage is concentrated in
a few periods, which is why the lower bound is not above zero. Of the two
disputed SPY closes, 2020-03-12 was the entry price of a SPY trade held on
2020-03-13 (recorded return +5.9%); 2020-03-18 was not a traded close. They were
not checked against a second source.

Per the contract there is no re-run, no threshold change and no variant scored
on this data under this registration. The holdout was not read, no model was
fitted and no order was sent. Any follow-up needs a new pre-registered contract
and the owner's word.
