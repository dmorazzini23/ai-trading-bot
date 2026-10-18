# Paper round-trip execution cost evidence (October 18, 2026)

This is read-only `evidence_reconciliation` work under the
[research reset](RESEARCH_RESET.md). It measures what the paper account actually
paid against the decision-time quote, compares that with the 10 bps round-trip
cost the three registered trials used, and states the fee gap exactly. No config,
policy, model, gate or trading code was changed. No trial was rerun. No order was
sent. No fee was imputed and no account-level charge was allocated to a fill.

Machine-readable output: `artifacts/cost_evidence_20261018/cost_evidence.json`
(sha256 `76c6ab17…2c66aa`). The generating script is
`artifacts/cost_evidence_20261018/build_cost_evidence.py` (sha256 `981e1db7…17f878`).
Both are in the main checkout's git-ignored `artifacts/` tree, alongside the trial
reports. Input file hashes are recorded in the JSON.

## Bottom line

- **Measured paper cost is near zero, and it understates live cost.** Across
  75 paired paper round trips (AAPL, AMZN, MSFT; 34 sessions, July 20 to
  October 16, 2026), the cost measured against the decision-time quote midpoint
  was: mean **−2.08 bps**, median **−0.71 bps**, p75 **+1.16 bps**, p95
  **+5.10 bps**. A negative value means the fill beat the mid. The bootstrap 95 %
  interval for the mean is [−3.69, −0.42] bps; for the median it is
  [−1.77, +0.03] bps. Paper fills come from Alpaca's simulator. It leaves out
  market impact, queue position and latency slippage, so these numbers are a
  **lower bound** on live cost, not an estimate of it.
- **A taker benchmark is higher.** A round trip that crosses the full
  decision-time spread pays the quoted spread: median **2.80 bps**, p75
  **5.63 bps**, p95 **25.3 bps** (MSFT median 5.16 bps, AMZN 2.71, AAPL 2.10).
- **No trial passes its screen at any of these costs.** Lower costs flip the sign
  of several folds and aggregates. But in every case the bootstrap lower bound
  stays negative, or the advantage over always-long goes away. The 10 bps
  assumption did not cause the rejections.
- **The fee gap is unchanged.** No fill record carries a fee. The paper account
  posts only daily account-level TAF, CAT and REG charges, with no execution
  reference. Alpaca's paper-trading page says paper trading does not account for
  regulatory fees.

## 1. Effective cost at fill versus the decision-time quote midpoint

**Method.** Partial fills are combined into one VWAP row per broker order. For
each order, per-side cost = `sign × (VWAP − mid) / mid × 10⁴`, where sign is +1
for buys and −1 for sells, so a positive value means cost was paid. The quote is
taken from the first of these sources that is available:

1. the decision record's TCA arrival bid and ask, joined by `client_order_id`
   (137 orders);
2. the decision quote stored on the fill event (8 orders);
3. for orders with no decision quote (end-of-day and closing market exits), the
   nearest prior per-bar decision-record quote for the same symbol, at most
   120 s before the fill (21 orders). This is a proxy, and it is reported
   separately.

Coverage: 166 of 7,510 fill orders have a quote. All 166 are from July 20 to
October 16, 2026, for AAPL, AMZN and MSFT, at 1–2 shares each (about $359 per
order). Of the 153 orders in `decision_records.jsonl` (decision bars
August 11 to October 16), 137 have a fill event. Unfilled limit orders cannot be
measured, which is a selection effect.

**Per side (bps, positive = paid)**

| Slice | n | Mean | Median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All quoted orders | 166 | −0.70 | 0.16 | 0.98 | 3.91 |
| Direct decision quote only (sources 1–2) | 145 | −0.61 | 0.30 | 0.95 | 2.61 |
| Buy | 89 | −1.23 | 0.00 | 0.87 | 1.82 |
| Sell | 77 | −0.09 | 0.30 | 1.31 | 17.09 |
| Limit | 145 | −0.61 | 0.30 | 0.95 | 2.61 |
| Market (all proxy quote) | 21 | −1.35 | −1.31 | 1.47 | 17.04 |
| Opening | 21 | −1.45 | −0.74 | 0.87 | 1.58 |
| Midday | 81 | −0.01 | 0.60 | 1.18 | 3.91 |
| Closing | 64 | −1.34 | −0.09 | 0.77 | 3.91 |
| AAPL | 53 | −0.68 | 0.30 | 0.90 | 2.65 |
| AMZN | 63 | −1.22 | −0.19 | 0.78 | 1.60 |
| MSFT | 50 | −0.07 | 0.32 | 1.59 | 20.67 |

**By symbol, side, session regime and order type**

| Symbol | Side | Session | Type | n | Mean | Median | p75 | p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AAPL | buy | opening | limit | 8 | −2.84 | −0.52 | 0.88 | 1.09 |
| AAPL | buy | midday | limit | 14 | −0.17 | 0.45 | 0.84 | 1.51 |
| AAPL | buy | closing | limit | 5 | −0.85 | −0.63 | 0.46 | 0.48 |
| AAPL | sell | opening | limit | 2 | 0.75 | 0.75 | 0.83 | 0.88 |
| AAPL | sell | midday | limit | 17 | −0.45 | 0.29 | 0.61 | 1.98 |
| AAPL | sell | closing | market | 7 | −0.04 | 1.31 | 2.82 | 13.18 |
| AMZN | buy | opening | limit | 7 | −2.40 | −0.96 | −0.47 | 1.05 |
| AMZN | buy | midday | limit | 8 | −0.13 | 0.00 | 0.83 | 1.11 |
| AMZN | buy | closing | limit | 19 | −0.25 | 0.36 | 0.78 | 1.02 |
| AMZN | sell | midday | limit | 13 | 0.11 | 0.80 | 1.01 | 2.53 |
| AMZN | sell | closing | limit | 6 | −1.10 | 0.01 | 0.48 | 0.59 |
| AMZN | sell | closing | market | 10 | −4.92 | −5.41 | −0.89 | 4.70 |
| MSFT | buy | opening | limit | 3 | −8.62 | −6.95 | −3.38 | −0.53 |
| MSFT | buy | midday | limit | 18 | −2.07 | 0.35 | 1.30 | 2.47 |
| MSFT | buy | closing | limit | 7 | 0.76 | 0.70 | 1.40 | 1.91 |
| MSFT | sell | opening | limit | 1 | 33.48 | 33.48 | 33.48 | 33.48 |
| MSFT | sell | midday | limit | 11 | 4.20 | 1.32 | 3.99 | 20.39 |
| MSFT | sell | closing | limit | 6 | −7.79 | −7.93 | −1.17 | 1.35 |
| MSFT | sell | closing | market | 4 | 5.28 | −0.51 | 5.86 | 19.93 |

Most cells hold fewer than 20 orders and should not be read on their own.

**Context for these numbers**

- Decision-time quoted spread: median 2.80 bps, mean 6.55, p75 5.63, p95 25.3,
  maximum 113.
- Fill versus the decision-time touch (ask for buys, bid for sells): median
  −1.16 bps and p95 0.00. Nearly every fill was at or inside the decision touch.
  That is consistent with
  [Alpaca's paper rules](https://docs.alpaca.markets/us/docs/paper-trading):
  orders fill only once they are marketable, and order size is not checked
  against NBBO size.
- Time from quote to fill: median 12.8 s, p75 100 s, p95 131 s. Cost measured
  against the decision mid therefore includes price drift over that interval as
  well as spread. The data holds no fill-time quote that would separate the two.
- Execution profile: 154 orders were `passive` and 12 `balanced`. In this
  sample, 53 limit sells are labelled `order_role: entry`. This is the
  closing-sell mislabel that `f03bc15a9` fixed for new records only; historical
  records are unchanged. The pairing below does not use the label.

## 2. Round trips

Pairing is FIFO, buy then sell, per symbol per UTC day, across the 166 quoted
orders. Round-trip cost is the entry cost plus the exit cost, each measured
against its own decision mid.

| Round-trip set | n | Mean | Median | p75 | p95 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All paired | 75 | −2.08 | −0.71 | 1.16 | 5.10 | −19.5 | 23.5 |
| Limit exit, direct quote | 54 | −2.10 | −0.48 | 1.13 | 3.63 | −19.5 | 23.5 |
| Market exit, proxy quote | 21 | −2.03 | −2.46 | 1.80 | 16.41 | −18.2 | 16.5 |
| AAPL | 25 | −1.44 | −0.59 | 1.16 | 5.39 | | |
| AMZN | 29 | −2.61 | −1.01 | 0.80 | 3.13 | | |
| MSFT | 21 | −2.10 | −0.71 | 2.41 | 16.49 | | |

A second estimate adds the per-side statistics together instead of pairing
orders. Buy plus sell gives a mean of −1.32 bps and a median of +0.30 bps. Adding
the two per-side p95 values gives 18.9 bps, an upper envelope driven by the
MSFT sell tail.

## 3. Realized slippage in the slippage logs and execution-quality events

- **`/var/log/ai-trading-bot/slippage.csv`** has 693 rows but only two distinct
  payloads: `AAPL buy 100 @150→151` (350 rows) and `MSFT buy 10 @100→101`
  (343 rows). Timestamps run from 2024-01-02 to 2026-04-06. These are test
  fixtures written into the production log. They contain **no execution
  evidence**.
- **Repository `logs/slippage.csv`** holds one fixture row
  (`AAPL unknown 1 @100→100`).
- **`execution_quality_events.jsonl`** has 58,263 rows: 44,709 `submit_skipped`,
  11,043 `cycle_kpi_snapshot`, 2,312 `submit_outcome` and 199 `submit_failed`.
  **No event type carries a slippage field.** The `submit_outcome` events carry
  `realized_net_edge_bps`, which is a post-trade outcome rather than a cost
  (n 2,264, median +0.53, mean +10.3). They also carry `realization_gap_bps`
  (median −3.89). `liquidity_role` is `unknown` or missing on every row.
- **`fill_events.jsonl` `slippage_bps`.** On all 7,462 orders where both values
  exist, the recorded value has the same magnitude and sign as cost measured
  against `expected_price` (positive = paid). But `expected_price` is a signal or
  reference price, not a quote mid. For fills with no quote (March to July,
  7,317 orders), cost against `expected_price` was: mean −5.63, median −1.47,
  p75 0.00, p95 2.87 bps. By month the median was March −1.85, April −1.14,
  May +0.18, June −1.01 and July +0.26 bps. April's mean of −8.76 bps is driven
  by tails of ±200 bps or more. This tier does not measure spread paid.
- **`tca_records.jsonl`** has 18,947 rows. Of these, 6,217 have a fill and an
  implementation-shortfall value. `is_bps` (positive = paid; verified to share
  sign with fill against arrival on 5,072 nonzero rows) has median −1.57 and
  mean −9.48. `spread_paid_bps` has median 0.00 and mean 2.34. No row has
  non-zero `fees`, and that is an absence, not a measurement of zero.

## 4. Runtime cost model and TCA state

| Source | What it says |
| --- | --- |
| `execution_cost_model.json` (`calibrated-20261009200006`, 91 samples) | `base_cost_bps` 2.0, which equals `min_bps` 2.0, so the base sits on its floor. `max_bps` 25. Weights: spread 0.423, participation 0.789, volatility 0.25, TCA 0.45. |
| `tca_feedback.json` (updated 2026-10-09 20:00 UTC, `sample_count` 1,500) | `total_bps`, `spread_bps`, `impact_bps` and `timing_bps` are all 0.0. Every per-symbol and per-hour median is 0.0 except BLK at 0.61. `target_total_bps` 12.0. |
| `live_cost_model_latest.json` (generated 2026-10-09) | Only two buckets, AAPL and MSFT midday buys, 3 samples each, `sufficient_samples: false`. Mean total cost 3.05 and 3.87 bps; p90 total 4.59 and 4.05 bps. |

The runtime model's 2.0 bps is a floor, not a measured central value. The TCA
feedback's zeros come from paper fills at or better than reference. Neither one
is independent evidence of live cost.

## 5. The fee gap, exactly

**What the local fill records hold.** All 7,510 rows have `fee_amount` and
`fee_bps` null. `fee_source` is `"missing"` on 126 rows and absent on 7,384.

**What the paper account activity API returns.** These figures come from the
existing paper snapshot fetched 2026-09-09 00:39 UTC
(`artifacts/research_reset/account_activities.json` and
`fee_followthrough.json`). No new account fetch was made.

- `FILL` rows (446) carry `activity_type`, `cum_qty`, `id`, `leaves_qty`,
  `order_id`, `order_status`, `price`, `qty`, `side`, `symbol`,
  `transaction_time` and `type`. **None carries a fee field.** That matches the
  documented TradeActivity object in Alpaca's
  [account activities contract](https://docs.alpaca.markets/us/docs/account-activities).
- `FEE` rows (81) carry `activity_sub_type`, `created_at`, `currency`, `date`,
  `description`, `id`, `net_amount` and `status`. **None carries an order or fill
  reference.** By sub-type:
  - TAF: 34 rows, −$0.34
  - CAT: 34 rows, −$0.34
  - REG: 13 rows, −$0.19

  The total is −$0.87 from 2026-06-10 to 2026-09-01. Each row is a daily
  aggregate rounded up to $0.01 or $0.02, for example
  "TAF fee for proceed of 4 shares (4 trades) on 2026-06-10" or
  "REG fee for proceed of $278.15 on 2026-08-10".
- For context only, and not as a per-fill fee: −$0.87 against the
  $207,719 of locally recorded fill notional in the same window is 0.042 bps.
  Under the reset this ratio must not be allocated to fills or used as a fee
  input. Penny rounding of daily totals dominates at 1–2 share order sizes.

**What Alpaca documents about paper fees.**
[Paper Trading](https://docs.alpaca.markets/us/docs/paper-trading) says that
paper trading does not account for *regulatory fees*, market impact, order queue
position for non-marketable limit orders, price slippage due to latency, or price
improvement received. It also says that partial fills occur at a random size
10 % of the time. The account nevertheless posts daily TAF, CAT and REG `FEE`
activities. That contradicts the documentation, so it is unclear what those rows
represent for a paper account, and they cannot be read as the complete charge.

[Regulatory Fees](https://docs.alpaca.markets/us/docs/regulatory-fees) says that
TAF applies to sells only and CAT to buys and sells. Total fees are rounded up to
the currency's precision and charged at end of day for each trading day. For
current rates the page points to the brokerage fee schedule at
<https://alpaca.markets/disclosures>. The fee schedule PDF there
(`BrokFeeSched.pdf`, metadata "Template Oct 12, 2026") could not be converted to
text here, and the secondary pages quote conflicting historical rates. This
report therefore states no numeric rate.

**What a declared fee schedule with explicit provenance would look like.** The
reset contract asks for broker-confirmed per-fill totals, and the paper API does
not provide them. The owner could instead accept a *declared* schedule. That is a
distinct evidence class, and it would need a reset amendment. A minimal record:

```json
{
  "schedule_id": "declared_us_equity_fees_v1",
  "evidence_class": "declared_schedule_not_broker_confirmed_per_fill",
  "declared_by": "<owner>",
  "declared_at": "<ISO-8601>",
  "reset_amendment": "<doc path authorizing use>",
  "source": {
    "url": "https://alpaca.markets/disclosures (BrokFeeSched.pdf)",
    "retrieved_at": "<ISO-8601>",
    "document_sha256": "<sha256 of the retrieved PDF>",
    "document_version_marker": "<e.g. PDF metadata title/date>"
  },
  "effective_from": "<date>", "effective_to": "<date or null>",
  "components": [
    {"name": "commission", "basis": "per_order", "rate": "<from schedule>", "sides": ["buy", "sell"]},
    {"name": "sec_section_31", "basis": "usd_per_usd_principal", "rate": "<from schedule>", "sides": ["sell"]},
    {"name": "finra_taf", "basis": "usd_per_share", "rate": "<from schedule>", "cap_usd_per_trade": "<from schedule>", "sides": ["sell"]},
    {"name": "cat", "basis": "usd_per_executed_equivalent_share", "rate": "<from schedule>", "sides": ["buy", "sell"]}
  ],
  "rounding": "round_up_to_0.01_usd_per_account_trading_day (per Alpaca Regulatory Fees page)",
  "application": "modeled_fee column computed per fill, kept separate from measured fee fields; never written into fee_amount",
  "not_covered": ["borrow fees", "exchange/ECN access fees or rebates", "penny-rounding allocation"],
  "promotion_authority": false
}
```

Rules for using such a schedule:

- Modeled fees stay in their own column and never fill `fee_amount` or
  `fee_source`.
- Every rate is copied from a hashed source document, never from memory.
- Each component lists its side.
- The schedule never replaces broker totals once those become available.
- This report does not adopt the schedule. It only describes it.

## 6. What the trials look like at the measured cost

**Method.** These figures are read from the existing `report.json` files only.
Nothing was rerun, and the per-opportunity prediction files were not read.

- **Two classifier trials.** Holding selection fixed,
  `net(c) = net(10) + (10 − c) × selected / rows`, applied per fold and in
  aggregate. The bootstrap lower bound is shifted by the same amount. That shift
  is approximate, because the selection share varies across resamples.
- **Momentum study.** It is a fixed rule, so re-pricing is exact for its
  selection. It uses `gross_per_opportunity − c × selected / rows`.

**Cost points (bps round trip).**

| Cost point | bps |
| --- | ---: |
| Zero floor | 0 |
| Measured paired p75 | 1.16 |
| Runtime model base | 2.0 |
| Quoted-spread median (full cross) | 2.80 |
| Measured paired p95 | 5.10 |
| Quoted-spread p75 | 5.63 |

The measured mean (−2.08) and median (−0.71) are negative. Re-pricing at a
negative cost would credit a rebate that the paper simulator generated, so the
central estimate enters as the zero floor.

**Mean net bps per opportunity, with positive folds out of 5**

| Cost (bps) | Model replacement, Sept 21 | Day-sleeve 60 min, Oct 18 | Momentum 30 m, frozen |
| --- | ---: | ---: | ---: |
| 10 (as judged) | −3.197 (0/5) | −0.854 (1/5) | −4.224 (0/5) |
| 0 (floor) | +0.072 (4/5) | +0.282 (4/5) | +1.024 (4/5) |
| 1.16 (paired p75) | −0.306 (0/5) | +0.151 (3/5) | +0.416 (3/5) |
| 2.0 (runtime base) | −0.582 (0/5) | +0.055 (3/5) | −0.026 (2/5) |
| 2.80 (quoted median) | −0.842 (0/5) | −0.036 (3/5) | −0.444 (1/5) |
| 5.10 (paired p95) | −1.596 (0/5) | −0.297 (1/5) | −1.654 (1/5) |
| 5.63 (quoted p75) | −1.770 (0/5) | −0.358 (1/5) | −1.933 (1/5) |
| Gross break-even per selection | 0.22 | 2.48 | 1.95 |

**Sign changes compared with 10 bps**

| Trial | Cost point | Folds that change sign | Aggregate changes sign | Selected opportunities in changed folds |
| --- | --- | ---: | :---: | ---: |
| Model replacement | 0 bps | 4 of 5 | yes | 26,267 of 30,802 |
| Model replacement | 1.16 bps or more | 0 | no | 0 |
| Day-sleeve | 0 bps | 3 of 5 | yes | 589 of 845 |
| Day-sleeve | 1.16 or 2.0 bps | 2 of 5 | yes | 412 of 845 |
| Day-sleeve | 2.80 bps | 2 of 5 | no | 412 of 845 |
| Day-sleeve | 5.10 bps or more | 0 | no | 0 |
| Momentum | 0 bps | 4 of 5 | yes | 1,049 of 1,235 |
| Momentum | 1.16 bps | 3 of 5 | yes | 819 of 1,235 |
| Momentum | 2.0 bps | 2 of 5 | no | 549 of 1,235 |
| Momentum | 2.80 bps or more | 1 of 5 | no | 298 of 1,235 |

**Why no trial passes its screen at any measured cost**

| Trial | At 0 bps | At 2.0 bps |
| --- | --- | --- |
| Model replacement | Bootstrap lower bound ≈ −0.18. Improvement over always-long falls from +6.75 to +0.02. | Mean −0.58, lower bound ≈ −0.83. |
| Day-sleeve | Lower bound ≈ −0.35. Improvement over always-long turns **negative** (−0.15). Four of five folds still lack the 250 selections the screen requires, whatever the cost. | Lower bound ≈ −0.58. |
| Momentum | Momentum minus always-long is −0.05. It is not better than holding. | Mean −0.03. |

Lower costs help always-long more than the selective strategies, because
always-long trades every opportunity. Most of the trials' apparent advantage at
10 bps was cost they avoided by not trading, not predictive edge.

**Dates.** The Sept 21 and Oct 18 trials cover 2024-05-06 to 2025-12-31. The
momentum study's folds cover **2026-03-27 to 2026-07-15**, not 2024–2025 (see
`artifacts/momentum_30m_study_frozen/metadata.json`). Momentum was registered at
6 bps, where it scored −2.125.

**Not answered: how many individual opportunities flip.** An opportunity's sign
flips when its gross return lies between the new cost and 10 bps. Counting those
needs the per-opportunity gross values in `oof_predictions.parquet` and
`opportunities.csv`. That is reading, not rerunning, but it was outside this
brief.

**Caveat on the classifier trials.** Their labels were defined as `gross > 10`.
At another cost, the labels, the fitted model and the selection would all
change. Re-pricing with selection held fixed shows only how the 10 bps choice
affected the score of the trades that were actually selected. It does not show
what a trial designed for that cost would have found.

## 7. What this measurement cannot establish

- **Live cost.** Paper fills leave out market impact, queue position, latency
  slippage and price improvement, and order size is never checked against
  displayed size
  ([Alpaca paper trading](https://docs.alpaca.markets/us/docs/paper-trading)).
  The negative mean is a simulator artifact combined with quote drift, not a
  rebate the strategy could earn.
- **Spread versus drift.** With a median of 12.8 s and a p95 of 131 s between
  quote and fill, and no fill-time quote, spread cost cannot be separated from
  price movement.
- **Non-fills.** Unfilled passive orders (16 of 153 in the decision records) are
  missing. Their opportunity cost is not measured.
- **Scale and coverage.** The sample is 75 round trips over 34 sessions in three
  symbols, at 1–2 shares. It says nothing about size, other symbols, opening
  auctions or volatile regimes beyond what was sampled. Most of the
  symbol × side × session cells hold fewer than 10 orders.
- **Proxy exits.** The 21 market exits use a quote from up to 120 s earlier.
- **Fees.** No per-fill total fee exists, and nothing here establishes zero
  fees.
- **Holdout and promotion.** The September 9 to December 8 holdout was not
  touched, and this report carries no promotion or runtime authority.

## AGENTS.md reporting

- **WORKLOG:** read-only evidence reconciliation of paper execution cost against
  the trials' fixed 10 bps. The root question is whether the cost assumption,
  rather than the lack of edge, drove the rejections. Answer: it did not.
- **PATCHSET:** this document only. The JSON and script are written to the
  git-ignored `artifacts/cost_evidence_20261018/`.
- **REGRESSION COVERAGE:** none. No runtime or library code changed.
- **VALIDATION:** `bash scripts/agent_validate_changed.sh --docs-only`.
- **RUNTIME CHECK:** skipped. There was no runtime change, and the service was
  not touched.
- **RISK & ROLLBACK:** documentation only. Revert the commit, and delete the
  artifact directory if it is not wanted.
