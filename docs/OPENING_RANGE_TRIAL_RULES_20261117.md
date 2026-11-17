# Opening-range breakout trial: the rules in plain words

**Status, November 17, 2026:** a draft for the owner to approve or reject before
any result exists. Nothing is registered. There is no ledger. No trade result
has been computed on real prices. The only contact with the data provider was a
small check of what data exists (see "What the data check found"). The full
contract is `config/opening_range_trial_campaign.json`; the program that would
score it is `ai_trading/tools/opening_range_trial.py`. It never places orders.

## The idea

Some mornings a stock trades far more than usual in its first five minutes.
Usually there is news. Day traders call these "stocks in play". A 2024 paper
(Zarattini, Barbon and Aziz) says that on such a morning the direction of the
first five minutes tends to continue for the rest of the day. It reports a
gain of about 41% a year from 2016 to 2023 for a simple rule that trades the
20 busiest such stocks each day. The same rule on all stocks, without the
"busiest" step, made about 3% a year.

## What is bought and sold, and when

**Which stocks can be considered (checked every morning):**

- The stock opens above $5.
- It traded at least 1,000,000 shares a day on average over the last 14
  trading days.
- Its typical daily price range over the last 14 days is above $0.50. (The
  paper calls this the average true range, ATR.)

**Which 20 are picked:**

- At 9:35 am New York time, look at each stock's volume from 9:30 to 9:35.
- Divide it by that stock's own average 9:30–9:35 volume over the last 14
  days. This is its "relative volume". A value of 2 means twice the usual.
- Drop any stock below 1 (less than usual).
- Take the 20 highest. On a quiet day there may be fewer than 20.
- Nothing that happens after 9:35 is used to pick.

**The order for each picked stock:**

- If its price rose between 9:30 and 9:35: place an order to **buy** if the
  price later goes above the highest price of those five minutes.
- If its price fell: place an order to **sell short** if the price later goes
  below the lowest price of those five minutes. (Selling short means selling
  borrowed shares to buy them back later, hoping the price falls.)
- If the price ended exactly where it began: no order.
- Many orders never trigger. Those cost nothing.

**Getting out:**

- Each trade has a safety exit, called a stop loss. It sits one tenth of the
  stock's typical daily range away from the entry price. That is close. For a
  $50 stock that typically moves $2 a day, it is 20 cents.
- If the stop loss is not hit, the trade is closed at the end of the day
  (4:00 pm, or 1:00 pm on a half day). Nothing is held overnight.

**How much:**

- The account is split into 20 equal slices, one per stock.
- Each trade is sized so that hitting the stop loss costs 1% of its slice.
  That is 0.05% of the account.
- No position may be larger than 4 times its slice (20% of the account). All
  20 together can be up to 4 times the account. That is borrowed money, and
  the broker allows it only within the day.

**Two versions are scored:**

- **Buy-only (the one that counts).** Only the buy orders. The paper account
  cannot sell short today, so this is the version that could be traded.
- **Buy and sell-short (secondary).** The rule as published. A short sale is
  skipped when it probably could not have been made: the broker's list says
  the stock is hard to borrow, or market rules restrict shorting that stock
  that day (they do after a 10% fall). This version cannot rescue a failed
  buy-only version.

## How fills are assumed

A "fill" is the price an order actually gets. The paper assumes every order
gets exactly the price it asked for. That is the weakest part of the paper,
because these are fast-moving stocks and the stop loss is very close.

Our price record has one line per minute: the first, highest, lowest and last
price of that minute. It does not show the order in which they happened. Every
doubt is settled against the strategy:

- **Entry.** We pay the worse of the price we asked for and the first price of
  the next minute.
- **Both in one minute.** If, in the minute the order triggers, the price was
  also low enough to hit the stop loss, we count it as a loss. We do this even
  if the low came first and the stock then ran in our favour. We cannot tell
  which came first.
- **Jumps.** If the price jumps past our level between two minutes, we get the
  price after the jump, not our level.
- **Stop loss.** We get the worse of the stop price and the first price of the
  next minute.
- **Trading halts.** If a stock stops trading more than 30 minutes before the
  close, we close the trade at its last price and then take off one full
  day's typical range. That is ten times the planned loss.

## What it costs

- **Commission:** $0.0035 per share, each way, as in the paper.
- **Slippage:** an extra charge on every buy and every sell, as a share of the
  trade's value. It stands for the gap between buying and selling prices and
  for chasing a moving stock.
  - **0.05% each way** decides the result.
  - **0.10% each way** is the stress case. The strategy must still make money.
  - 0.025% each way, and no slippage at all (the paper's assumption), are
    shown for information only.
- **Why 0.05%.** The bot's own orders in three giant stocks at midday cost
  about 0.01% on average and 0.06% in the worst tenth. The stocks here are
  smaller, cheaper and moving fast just after the open. We take the giant
  stocks' worst-tenth cost as the normal cost here. It is an assumption. It
  has not been measured for these stocks at this time of day.
- **Why this matters so much.** The stop loss is about 0.4% of the price for
  a typical stock. Buying and selling at the 0.05% level costs 0.10% of the
  price. So every trade, win or lose, gives up about a quarter of the planned
  loss in costs. The paper reports an average gain per trade of well under
  one planned loss. **A fail is a realistic outcome.**

## What counts as passing

Tested on January 2016 to December 2025: about 2,500 trading days.

**Buy-only passes only if all of these hold:**

1. It made money after all costs, by a margin that a standard resampling test
   separates from luck.
2. It still made money at double the slippage (0.10% each way).
3. It made money in at least 6 of the 10 calendar years.
4. It beat simply holding the same amount of the whole market (the SPY fund)
   over the same minutes, again by a margin the resampling test separates
   from luck. This shows the gain is not just the market rising that day.
5. There was enough data: at least 2,300 usable days, at least 230 in each of
   9 years, at least 10 stocks picked on a typical day, and at least 5,000
   completed trades.

The buy-and-sell-short version has tests 1, 2, 3 and 5.

**A pass** allows one final check on September 9 to December 8, 2026, which
has been kept unseen, and then at least eight weeks on the paper account. It
does not allow real-money trading. **A fail** ends the idea for both versions.
The rules are not adjusted and re-run.

## Known weaknesses

- **One group of authors, with something to sell.** The authors run
  businesses that sell day-trading tools and courses. We know of no
  independent test that used realistic fills.
- **A short sample.** The paper covers eight years. They include 2020 and
  2021, when small investors traded far more than usual. We add 2024 and 2025.
- **Fills.** The published result needs every order filled at exactly its
  price. Our rules lean the other way. They may be too harsh as well as too
  kind. One-minute prices cannot settle it. In particular the "both in one
  minute" rule may count many good trades as losses.
- **Missing dead companies.** See the data check below. Companies that were
  bought or went bankrupt are partly missing. Those are often the busiest
  stocks on their biggest days. We do not know whether leaving them out makes
  the result look better or worse.
- **Short sales.** We cannot know whether shares could really have been
  borrowed on a given morning in the past.
- **Borrowed money.** Trading at up to 4 times the account needs a margin
  account of at least $25,000 under US day-trading rules.
- **Account size.** Results are stated per dollar. A small account pays
  minimum commissions. A large account would move these stocks.
- **Eleven tries.** Nine earlier hypotheses failed and this adds two. The
  test is stricter to allow for that, but a first pass in eleven attempts
  still has roughly a 1-in-8 chance of being luck.
- **The window is not fresh.** Earlier research has already looked at some
  prices from these years.

## What the data check found

Run on November 17, 2026. It asked the provider what exists for a few stocks
on a few days. It kept counts and times only, no prices.

- **Minute-by-minute prices go back to the first day we need** (January 4,
  2016), for large and small stocks alike. A stock has no line for a minute
  with no trade; that is normal.
- **The 9:30–9:35 summary agrees with the five one-minute lines** in all 25
  cases checked.
- **Dead companies: the prices exist, the names do not.** For 21 large
  companies that were bought or failed (Twitter, Yahoo, LinkedIn, Monsanto,
  Bed Bath & Beyond, Silicon Valley Bank and others), the provider returned
  minute prices for all 21. But the broker's list of stocks, which is how we
  find candidates, contains only 10 of them. The other 11 would never be
  considered. The paper's stock list included them.
- **The download is workable.** About 12,000 requests. The provider allows
  200 a minute and the running bot shares that allowance, so the program uses
  about half. That is 2 to 3 hours, best done while the market is closed.
  About half a gigabyte on disk. These figures are estimates.
- **Why it is small.** We do not download every stock's whole day. Picking
  the 20 needs only each stock's first five minutes. Only the 20 picked each
  day need their minute-by-minute prices. That is about 2% of the full
  download and changes no trade.

## Where this differs from the paper

| Point | Paper | This test |
|---|---|---|
| Fills | Exactly the order price | Pessimistic rules above, plus slippage |
| Stock list | Research database including dead companies | Broker's list; many dead companies missing |
| Short sales | Always possible | Skipped when probably not possible; "always" shown for information |
| Version that counts | Buy and sell-short | Buy-only, because the account cannot short |
| Size of a trade | "1% of the capital allocated to the position", up to 4 times | Read as 1% of one twentieth of the account, up to 4 times that slice |
| Typical range (ATR) | "14-day ATR" | Plain average of the last 14 days |
| Stock splits | Not mentioned | Stock left out for 15 days from a split |
| Trading halts | Not mentioned | Closed at the last price less one day's range |
| Years | 2016–2023 | 2016–2025 |

The filters, the 14 days, the 20 stocks, the five minutes, the stop loss and
the commission are the paper's numbers, unchanged.

## Questions for the owner

1. **Size of a trade.** The request said "1% of equity". The paper's words fit
   "1% of the slice" better (see the table). Which is meant? It changes the
   scale of results and which trades are capped.
2. **Which version counts.** Buy-only (tradable today) or the published
   buy-and-sell-short?
3. **Slippage levels.** Are 0.05% / 0.10% / 0.025% the right base, double and
   half?
4. **The "both in one minute" rule.** It was asked for and it is strict. It
   may decide the result on its own. Is that intended?
5. **Missing dead companies.** Accept the gap with a warning label, or supply
   a list of delisted ticker symbols from another source? The prices are
   there if we know the names.

## Steps after approval

Each needs the owner's word.

```bash
RUN="./venv/bin/python -m ai_trading.tools.opening_range_trial --repository-root . --contract config/opening_range_trial_campaign.json"
# independent review; adopt an amendment; set status to registered; register the ledger
$RUN --acquire         # one download, 2 to 3 hours, market closed; needs credentials; never claims
$RUN --support-audit   # days, names and times only; never reads prices or claims
$RUN --preflight       # file hashes, date bounds, price validity, rebuilds the daily picks; never claims
$RUN --fit             # the single claim
```
