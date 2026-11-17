# Opening-range breakout trial: the rules in plain words

**Status, November 17, 2026:** approved by the owner and registered as one
trial ([amendment](RESEARCH_RESET_AMENDMENT_20261117_OPENING_RANGE.md)). The
rules below were fixed before any result existed. They were revised once, the
same day, after an independent review (see "What changed after review"), and
the owner's answers are recorded near the end. At registration there is no
ledger yet, nothing has been downloaded for scoring, and no trade result has
been computed on real prices. The only contact with the data provider was a
small check of what data exists (see "What the data check found"). The full
contract is `config/opening_range_trial_campaign.json`; the program that
scores it is `ai_trading/tools/opening_range_trial.py`. It never places orders.

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

- It is an ordinary US company share. Funds are left out. So are foreign
  companies' US-traded receipts (ADRs), warrants, units, rights and preferred
  shares; they are recognised by words in the stock's name, which is not
  perfect.
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
- A stock needs all 14 of its previous 9:30–9:35 volumes. If one is missing
  (a new listing, a day with no early trade), the stock is not ranked until
  it has 14 in a row again.
- Take the 20 highest. On a quiet day there may be fewer than 20.
- Nothing that happens after 9:35 is used to pick.

**The order for each picked stock:**

- If its price rose between 9:30 and 9:35: **buy** once the price later goes
  above the highest price of those five minutes.
- If its price fell: **sell short** once the price later goes below the
  lowest price of those five minutes. (Selling short means selling borrowed
  shares to buy them back later, hoping the price falls.)
- If the price ended exactly where it began: no order.
- Many orders never trigger. Those cost nothing.

**Getting out:**

- Each trade has a safety exit, called a stop loss. It sits one tenth of the
  stock's typical daily range away from the price actually paid, not from the
  price the order was aiming at. That is close. For a $50 stock that
  typically moves $2 a day, it is 20 cents.
- If the stop loss is not hit, the trade is closed at the end of the day
  (4:00 pm, or 1:00 pm on a half day). Nothing is held overnight.

**How much:**

- The account is split into 20 equal slices, one per stock.
- Each trade is sized so that hitting the stop loss costs 1% of its slice.
  That is 0.05% of the account.
- No position may be larger than 4 times its slice (20% of the account). All
  20 together can be up to 4 times the account. That is borrowed money, and
  the broker allows it only within the day.
- For information only, the report also shows every trade at the full 20% of
  the account. That is what the other way of reading the paper's sizing
  sentence comes to. It decides nothing.

**Two versions are scored:**

- **Buy-only (the one that counts).** Only the buys. The paper account cannot
  sell short today, so this is the version that could be traded. It is the
  only decision.
- **Buy and sell-short (information only).** The rule as published. A short
  sale is skipped when it probably could not have been made: the broker's
  list says the stock is hard to borrow, or market rules restrict shorting
  that stock that day (they do after a 10% fall). **If this version passes
  while buy-only fails, nothing is permitted.** The idea is still retired. The
  report only notes that the published form showed something.

## How fills are assumed

A "fill" is the price an order actually gets. The paper assumes every order
gets exactly the price it asked for. That is the weakest part of the paper,
because these are fast-moving stocks and the stop loss is very close.

Our price record has one line per minute: the first, highest, lowest and last
price of that minute. It does not show the order in which they happened
inside the minute. The first draft settled every doubt against the strategy.
An independent review showed that this was not caution but a built-in loss:
on made-up prices with no edge at all, those rules lost 0.2 to 0.6 of the
planned loss on every trade before any cost. No strategy could pass.

**The fill rule that decides** uses only prices whose order we know:

- **Entry.** Wait for the first minute in which the price goes through the
  level. Then buy at the first price of the *next* minute, whatever it is.
  It may be above or below the level. (If the level is reached only in the
  day's last minute, there is no trade.)
- **Stop loss.** Measured from the price paid. If a later minute's lowest
  price reaches it, we sell at the stop price. If a minute *opens* already
  beyond it, we sell at that opening price, which is worse.
- **Otherwise** we sell at the last price of the day.
- **Trading halts.** If a stock stops trading more than 30 minutes before the
  close, we close the trade at its last price and then take off one full
  day's typical range (ten times the planned loss), but never below one cent.

On the same made-up prices with no edge, this rule comes out at about zero
(between −0.02 and +0.09 of the planned loss per trade across calm to very
jumpy prices). A program test now checks this every time the code changes.

**Three other fill rules are scored beside it, for information only:**

- **Cautious:** the same entry, but a stop-loss sale gets the worse of the
  stop price and the next minute's first price. On no-edge prices it loses
  about 0.1 to 0.2 of the planned loss per trade. It was first drafted as a
  pass condition; the owner made it information only, because it could fail a
  real edge by itself.
- **The paper's own** (the best case): every order filled at exactly its
  price.
- **The first draft's** (the worst case).

## What it costs

- **Commission:** $0.0035 per share, each way, as in the paper.
- **Slippage:** an extra charge on every buy and every sell, as a share of the
  trade's value. It stands for the gap between buying and selling prices and
  for chasing a moving stock. It is charged once; the fill rule above adds
  none of its own.
  - **0.05% each way** decides the result.
  - **0.075% each way** is the stress case. The strategy must still make money.
  - 0.10%, 0.025% and no slippage at all (the paper's assumption) are shown
    for information only.
- **Why 0.05%.** The bot's own orders in three giant stocks at midday cost
  about 0.01% on average and 0.06% in the worst tenth. The stocks here are
  smaller, cheaper and moving fast just after the open. We take the giant
  stocks' worst-tenth cost as the normal cost here. It is an assumption. It
  has not been measured for these stocks at this time of day.
- **The arithmetic, stated before any result.**
  - A typical position is 10% to 12.5% of the account.
  - Buying and selling it at 0.05% each way costs 2 × 0.05% × 10–12.5% =
    0.010% to 0.0125% of the account per trade.
  - With 12 to 15 trades a day, that is 0.12% to 0.19% of the account a day.
    Call it **0.15% a day in costs.**
  - The paper's published gain, 41.6% a year, is **0.14% a day**, and that is
    with perfect fills and no slippage.
  - So this cost alone is about as large as everything the paper says the
    strategy earns. **Even if the paper is right about what it measured, a
    fail at this cost is the likely outcome.** A pass would need real fills
    to be better than the paper assumed, or costs to be lower than we assume.

## What counts as passing

Tested on January 2016 to December 2025: about 2,500 trading days.

**Buy-only passes only if all of these hold:**

1. It made money after all costs, by a margin that a standard resampling test
   separates from luck.
2. It still made money at one and a half times the slippage (0.075% each way).
3. It made money in at least 6 of the 10 calendar years.
4. It beat simply holding the same amount of the whole market (the SPY fund)
   over the same minutes, again by a margin the resampling test separates
   from luck. This shows the gain is not just the market rising that day.
5. There was enough data: at least 2,300 usable days, at least 230 in each of
   9 years, at least 10 stocks picked on a typical day, and at least 5,000
   completed trades.

The buy-and-sell-short version is measured against tests 1, 2, 3 and 5,
for information. Its result permits nothing.

**Days that are left out.** A day is used only if the market fund SPY has a
price for every single minute of it. A gap means our record of that day is
incomplete, or trading was stopped for the whole market. This leaves out the
four days in March 2020 when the whole market was halted, which were among
the wildest days in the ten years. The first 15 days of 2016 only supply
history.

**A buy-only pass** allows one final check on September 9 to December 8, 2026, which
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
  price. Our deciding rule buys one minute later, at whatever the price then
  is. That is not the trade the paper describes: sometimes it is a better
  price, often a worse one. One-minute prices cannot settle what a real order
  would have got. The best case and the worst case are shown beside it.
- **Costs are probably as large as the edge.** See the arithmetic above.
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
- **Could the dead companies be added?** Probably, partly. Looked into, not
  built. Free lists of delisted ticker symbols with dates exist from data
  vendors (for example Alpha Vantage's listing-status file). The official
  record, the SEC's delisting filings, names the company but not reliably its
  ticker. Adding a vendor list would need one more file, frozen like the rest,
  and one more decision by the owner. Gaps would remain: the lists cannot be
  checked for completeness, and some tickers were later reused by other
  companies. We did not check what the lists contain.
- **The download is workable.** About 12,000 requests. The provider allows
  200 a minute and the running bot shares that allowance, so the program uses
  about half. That is 2 to 3 hours, best done while the market is closed.
  About half a gigabyte on disk. These figures are estimates.
- **When the download is refused.** The program stops, and keeps nothing, if:
  more than 1% of the stocks asked for fail to download in the monthly or
  daily step; on any single day more than 1% of that day's 9:30–9:35 requests
  fail, or fewer than 90% of the stocks asked for come back; more than 1% of
  all orders have no minute-by-minute prices; or more than 1% of days have an
  incomplete SPY record. One bad day cannot hide among 2,500 good ones.
- **Memory.** The machine has little spare memory beside the running bot. The
  step that picks the daily 20 was rewritten to use about a third of what it
  did; on made-up data at full size it peaked at 515 MB.
- **Why it is small.** We do not download every stock's whole day. Picking
  the 20 needs only each stock's first five minutes. Only the 20 picked each
  day need their minute-by-minute prices. That is about 2% of the full
  download and changes no trade.

## Where this differs from the paper

| Point | Paper | This test |
|---|---|---|
| Fills | Exactly the order price | Buy at the next minute's first price; stop measured from the price paid; slippage charged once on top |
| Stock list | Research database including dead companies | Broker's list; many dead companies missing |
| Short sales | Always possible | Skipped when probably not possible; "always" shown for information |
| Version that counts | Buy and sell-short | Buy-only, because the account cannot short; buy-and-sell-short permits nothing |
| Size of a trade | "1% of the capital allocated to the position", up to 4 times | Read as 1% of one twentieth of the account, up to 4 times that slice; the other reading (every trade 20% of the account) shown for information |
| Typical range (ATR) | "14-day ATR" | Plain average of the last 14 days |
| Stock splits | Not mentioned | Stock left out for 15 days from a split |
| Trading halts | Not mentioned | Closed at the last price less one day's range |
| Years | 2016–2023 | 2016–2025 |

The filters, the 14 days, the 20 stocks, the five minutes, the stop loss and
the commission are the paper's numbers, unchanged.

## The owner's answers (November 17, 2026)

The owner approved the rules "as recommended".

1. **Size of a trade.** As drafted: 1% of the slice decides; the other
   reading is shown for information.
2. **The cautious fill rule.** For information only. It is no longer a pass
   condition for either version.
3. **Cost levels.** Run it: 0.05% each way decides, 0.075% is the stress
   case, in the knowledge that a fail is the likely outcome.
4. **Missing dead companies.** Accept the gap with the warning label. No
   vendor list is added.
5. **The trading-halt rule and the short-sale restriction.** Stand as
   drafted.

## What changed after review

- The deciding fill rule: was "worst case inside the minute", now "buy at the
  next minute's first price". The old rule is kept as the worst case, for
  information.
- A cautious fill rule and the paper's exact fills were added. The cautious
  rule was first a pass condition; by the owner's decision it is information
  only.
- The stress cost: was 0.10% each way, now 0.075%; 0.10% is shown only.
- The download now checks every day on its own.
- The step that picks the daily 20 uses far less memory.
- A halted stock's closing price can no longer go below one cent.
- Buy-and-sell-short passing alone now explicitly permits nothing.
- A ticker spelled "NA" is no longer misread as a blank.

## Steps from here

Done: review findings closed, rules approved, amendment adopted, contract
marked registered. Each remaining step needs the owner's word.

```bash
RUN="./venv/bin/python -m ai_trading.tools.opening_range_trial --repository-root . --contract config/opening_range_trial_campaign.json"
# create the ledger (artifacts/opening_range_trial_20261117/campaign_state.json) before any price is read for scoring
$RUN --acquire         # one download, 2 to 3 hours, market closed; needs credentials; never claims
$RUN --support-audit   # days, names and times only; never reads prices or claims
$RUN --preflight       # file hashes, date bounds, price validity, rebuilds the daily picks; never claims
$RUN --fit             # the single claim
```
