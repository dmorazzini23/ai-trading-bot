# Swing reversal trial: the rules in plain words

**Status, November 15, 2026:** a draft for the owner to approve or reject before
any result exists. Nothing is registered. No price history has been downloaded
and nothing has been scored. The full contract is
`config/swing_reversal_trial_campaign.json`; the program that would score it is
`ai_trading/tools/swing_reversal_trial.py`. It never places orders.

## The idea

When investors need to sell a large stock quickly, they push its price below
where it would otherwise trade. Whoever buys from them tends to be paid as the
price recovers over the next few days. Research finds this payment is larger
when markets are fearful, and that it survives trading costs only in big,
heavily traded stocks and only when the strategy avoids trading too often.

## What it buys and sells

- **Which stocks.** Each month, the 300 US stocks with the most dollars traded
  over the previous three months, priced at $5 or more. The list is rebuilt on
  the first trading day of each month from past data only.
- **The signal.** For each stock, its price change over the last five trading
  days minus the average change of the stocks in the same sector. A stock that
  fell 6% while its sector fell 5% is barely a loser; one that fell 6% while
  its sector rose is.
- **Long-only version (the one that counts).** Buy the 30 stocks that fell the
  most against their sector.
- **Long-short version (secondary).** The same 30, plus selling short the 30
  that rose the most against their sector. The paper account has shorting
  switched off, so this version cannot be traded today.

## When, and for how long

- The ranking is made after the close every fifth trading day.
- Trades happen at the **next morning's open**, never at the close that
  produced the ranking.
- Positions are held for five trading days. Then the ranking is redone.
- **Hold rule.** A stock already held is kept as long as it is still in the
  losing half of the ranking (winning half for shorts). Only stocks that leave
  that half are replaced, by the most extreme new names. This cuts trading
  roughly in half.
- **Size.** Each stock gets one thirtieth of the amount invested. The amount
  invested rises with recent market turbulence: fully invested when the S&P 500
  fund has been moving at 20% a year or more over the past month, down to one
  quarter invested when it is calm. No borrowing to invest more.

## What it costs

- Every purchase and every sale is charged, including small resizing trades.
  The test uses **5 hundredths of a percent per trade** to decide, and
  **10** as a stress case. The bot's own fills suggest about 2, but those were
  one-share midday orders in three giant stocks on a simulator, so 2 is
  reported for information only. A result that works at 2 but not at 5 does
  not count.
- Expected trading: about half of the holdings change every five days, about
  25 times the portfolio per year. At the decision cost that is about **2.5% a
  year** for the long-only version and about twice that with the short side.
- Short positions also pay an assumed borrowing fee of 1% a year (3% in the
  stress case). Stocks that are probably hard to borrow are not shorted.

## What counts as a pass

Tested on January 2016 to December 2025. **Long-only passes only if all of
these hold:**

1. It beat holding the same amount of the S&P 500 fund on the same days, after
   costs, by a margin that a standard resampling test separates from luck.
2. It made money after costs by the same test.
3. It still beat the S&P 500 fund at double the cost.
4. It beat the S&P 500 fund in at least 6 of the 10 years.
5. It still beat it after assuming a 30% loss on every holding whose price
   history stopped while held.
6. There was enough data: at least 400 five-day periods across at least 8
   years, usually with 200 or more stocks to rank.

The long-short version has the same tests against cash. It cannot rescue a
failed long-only version.

**A pass** allows one final check on September 9 to December 8, 2026, which
has been kept unseen. It does not allow trading. **A fail** ends the idea. The
rules are not adjusted and re-run.

## Known weaknesses

- **Missing dead companies.** If the data provider leaves out stocks that
  were later delisted, a buy-the-losers test looks better than it was. Whether
  it does is not yet known: the check could not be run, because it needs the
  account's data credentials and reading the file that locates them was not
  permitted. It must be run before registration. If stocks are missing, every
  result will be labelled as flattering.
- **It trades a lot.** Even with the hold rule, turnover is far above the
  level at which published strategies usually survive costs.
- **Costs are assumptions.** Nothing here measures fills at the open. The
  bot's quote feed is also IEX-only, so its recorded spreads are not the
  market-wide best prices. That does not affect this test, which uses daily
  prices, but it limits how far the measured cost can be trusted.
- **The effect is well known** and has weakened in large stocks. The
  supportive studies end before 2012 and assume institutional costs.
- **It pays mostly in frightened markets.** It can look poor for years, and
  it is most invested when losses are most likely.
- **Sector labels are today's**, not those in force at the time, and are
  often missing for companies that no longer trade.
- **The funds filter is approximate.** The provider does not say which
  listings are funds, so they are removed by exchange and by name.
- **Long-only results are mostly the market.** Only the comparison with the
  S&P 500 fund says anything about the signal.
- **Seven tries.** Five earlier hypotheses failed and this contract adds two.
  The test is stricter to allow for that, but a first pass in seven attempts
  still has roughly a 1-in-12 chance of being luck.

## Steps after approval

Each needs the owner's word. The order follows the contract's preconditions.

```bash
RUN="./venv/bin/python -m ai_trading.tools.swing_reversal_trial --repository-root . --contract config/swing_reversal_trial_campaign.json"
$RUN --probe           # which delisted stocks the provider serves; dates only; allowed on the draft
# record the probe result in the contract; adopt the amendment; set status to registered; register the ledger
$RUN --acquire         # one download; needs credentials; never claims
$RUN --support-audit   # dates only; never reads prices or claims
$RUN --preflight       # file hashes, date bounds, price validity; never claims
$RUN --fit             # the single claim, from a closed market window
```
