# Intraday momentum trial: the rules in plain words

**Status, November 15, 2026:** a draft for the owner to approve or reject before
any result exists. Nothing is registered. No price history has been downloaded
and nothing has been scored. The full contract is
`config/intraday_momentum_trial_campaign.json`; the program that would score it
is `ai_trading/tools/intraday_momentum_trial.py`. It never places orders.

## The idea

Two published studies found that the direction of the stock market's morning
tends to carry into its last half hour. The explanation offered is forced
trading: option dealers and leveraged funds have to buy late in the day when
the market is up and sell when it is down. The studies find it stronger when
markets are turbulent.

## What is traded, and when

- **What.** One thing only: SPY, the fund that tracks the S&P 500.
- **The signal.** At 10:00 New York time, compare SPY's price with yesterday's
  closing price. Up is a positive signal, down is a negative one. Nothing after
  10:00 is used. There is no minimum size and no other filter.
- **Long-only version (the one that counts).** On a positive signal, buy SPY at
  3:30 pm and sell at the 4:00 pm close. On a negative signal, do nothing.
- **Long-short version (secondary).** The same, plus selling SPY short at
  3:30 pm on a negative signal and buying it back at the close. The paper
  account has shorting switched off, so this version cannot be traded today.
- **Size.** All of the account while in the trade, with no borrowing. Nothing
  is ever held overnight.
- **Days left out.** Days when the market closes early (about two a year)
  and days with any gap in the price record are not traded. They are counted
  in the report.

## What it costs

- Each buy and each sell is charged **0.01%** of the position to decide, and
  **0.02%** as a stress case. So a traded day costs 0.02% (0.04% under stress).
- The bot's own fills in three giant stocks cost about 0.002% typically and
  0.009% on average, so 0.01% is on the cautious side of what was measured.
  None of those fills was in SPY or late in the day. A cheaper case, 0.005%, is
  shown for information only.
- If about half the days are traded, the long-only version pays roughly 2.5%
  of the account a year in costs. The long-short version trades every day and
  pays about 5%.

## What counts as a pass

Tested on January 2016 to December 2025, about 2,490 usable days. **Long-only
passes only if all of these hold:**

1. It made money after costs, by a margin that a standard resampling test
   separates from luck.
2. It still made money at double the cost.
3. It made money in at least 6 of the 10 years.
4. It did better than simply buying SPY at 3:30 pm every day at the same cost.
   This shows the morning signal matters, not just the time of day.
5. There was enough data: at least 2,300 usable days, at least 230 in each of
   9 years, and at least 800 days actually traded.

The long-short version has tests 1, 2, 3 and 5. It cannot rescue a failed
long-only version.

**A pass** allows one final check on September 9 to December 8, 2026, which has
been kept unseen, and then at least eight weeks on the paper account. It does
not allow real-money trading. **A fail** ends the idea for both versions. The
rules are not adjusted and re-run.

Two extra versions are shown for information and decide nothing: trading only
when the 3:00 to 3:30 pm move agrees with the morning, and trading only when
the past month has been more turbulent than usual.

## Known weaknesses

- **The effect is small.** Our notes put the published gain at roughly 0.02%
  to 0.03% per day before costs; that figure is from memory and has not been
  re-checked against the paper. The cost charged here is 0.02% per traded day.
  Most of such an edge would be spent on trading, so a fail is a realistic
  outcome.
- **The evidence is old.** The SPY study ends in 2013. Informal checks on later
  years found it weaker, and same-day options have since changed how dealers
  trade.
- **It is concentrated in turbulent markets.** The plain rule trades in calm
  markets too, which dilutes it. The turbulent-only version is shown but does
  not count.
- **The prices are not fills.** The test buys at the first price recorded at
  3:30 pm and sells at the last price before 4:00 pm. A real order would not
  get exactly those prices, and the closing auction (the single large trade
  that sets the official close) is not modelled.
- **Paper fills flatter.** The paper account ignores the queue of other orders
  and the effect of the bot's own order on the price.
- **Test 4 is partly a saving on costs.** Sitting out half the days saves
  trading costs whether or not the signal is right. The same comparison
  without costs is shown beside it. Test 4 also needs only a positive average,
  not the stricter resampling margin, because it rests on about half the days.
- **Nine tries.** Seven earlier hypotheses failed and this adds two. The test
  is stricter to allow for that, but a first pass in nine attempts still has
  roughly a 1-in-9 chance of being luck.
- **The window is not fresh.** Earlier research has already looked at SPY's
  daily prices for these years.

## Steps after approval

Each needs the owner's word. One download: SPY half-hour prices for ten years,
about 80,000 rows fetched and about 32,500 kept, a file of 2 to 3 MB.

```bash
RUN="./venv/bin/python -m ai_trading.tools.intraday_momentum_trial --repository-root . --contract config/intraday_momentum_trial_campaign.json"
# adopt the amendment; set status to registered; register the ledger
$RUN --acquire         # one download; needs credentials; never claims
$RUN --support-audit   # dates and times only; never reads prices or claims
$RUN --preflight       # file hash, date bounds, price validity; never claims
$RUN --fit             # the single claim, from a closed market window
```
