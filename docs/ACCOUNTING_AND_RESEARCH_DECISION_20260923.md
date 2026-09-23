# September 23 accounting boundary and research decision

This review uses the September 22 paper session and existing development
artifacts. It does not change trading, model, replay, cost, freshness,
provenance, or promotion gates. All timestamps below are UTC.

## September 22 execution records

The paper session has seven local `fill_events.jsonl` rows, seven matching
order quantities, and a matched flat opening-to-closing broker position
boundary (13:29:31 to 20:01:53). Two local rows lack a decision and a resolved
TCA row:

| Local fill | Broker order | Recorded cause | Missing evidence |
| --- | --- | --- | --- |
| `derived-1a7165163ddf9a434597220dd30fc834` AMZN sell 1 at 19:55:13 | `98d6fd5b-92bd-4291-b851-81a77f0b5981` | `eod-2026-09-22-AMZN-sell`; matched paper order has `order_role=exit` | Strategy decision and resolved TCA |
| `derived-0bab05526687d8e488ae4c3a3dfefba6` MSFT sell 2 at 19:55:15 | `f4bf1aa3-d621-48b6-b9ad-c56a680c23f2` | `eod-2026-09-22-MSFT-sell`; matched paper order has `order_role=exit` | Strategy decision and resolved TCA |

The `exit_all_positions` safety path submitted these reduction orders through
`execute_order`, outside the netting path that writes strategy decisions and
pending TCA. The order/fill link is supported by the same account, broker order
ID and client order ID. No strategy signal or arrival quote can be inferred from
that link. The reconciliation now reports these as *operational exits with
missing decision/TCA evidence*, without counting them as matched decisions or
resolved cost measurements. Explicit account, mode and symbol conflicts cannot
produce a source match.

The broker account-activity capture contains **eight** executions on September
22. MSFT's one local sell-2 row is an aggregate of two broker sell-1 execution
activities at 19:55:14.225077 and 19:55:15.576123 on the same order. The
broker-backed quantity rebuild includes all eight and matches the flat closing
position. The local order-quantity match alone therefore did not establish
one-to-one execution capture. The accounting comparator now reports execution
counts per order; an equal count is still not proof of individual identity.
The original local and broker records remain untouched. Use the broker-backed
execution rows for that session's quantity accounting; exclude the two
operational exits from strategy decision/TCA completeness claims. Do not treat
the sell-2 aggregate as a single broker execution.

**Status:** Account/order/quantity cause resolved; the two historical strategy
decision and TCA matches are unavailable because they were never recorded.
Next regular session: confirm the same-account order/exit classification and
broker activity count comparison after close. Acceptance for complete future
trade evidence requires a causal operational-intent record at submission,
execution-level broker identities, a genuine arrival benchmark for TCA, and
complete fee evidence. Those missing fields cannot be backfilled from this
session's later fill or order status without misrepresenting causality.

## Execution fees and net performance

The September 22 paginated paper activity snapshot has 419 historical `FILL`
rows and 73 historical `FEE` rows. All seven local September 22 fill rows have
unknown *verified total* fees. The broker `FILL` schema in the captured API
response contains price, quantity, order ID and execution time, but no fee
amount. The captured `FEE` rows carry `net_amount`, accounting date and a
description, but no fill or order reference. No `FEE` row is dated September 22
in the snapshot taken at 20:36; non-trade activities may post later. The
ingestion already retains these charges as account-level entries and accepts
per-fill totals only with explicit broker source, USD currency and
`per_fill_total` basis. There is no observed per-fill fee field for it to ingest
here, so assigning charges to the seven local rows would be invented data.

Broker-backed fees remain distinct from configured cost estimates and unknown
costs. Account-level charges may be reported by their own accounting date, but
cannot certify a fill's total fee or net P&L. Seven unknown totals mean this
session supports quantity and gross arithmetic only, not verified net
profitability. After broker posting, check for an execution-linked, complete
fee source. Acceptance for verified net reporting is per-execution totals (or
an authoritative complete per-order allocation contract) with account, order,
execution, amount, currency and total-fee semantics; otherwise retain unknown
net P&L. Do not infer zero from absent fee activity.

Alpaca's newer [Activity SSE](https://docs.alpaca.markets/us/docs/activity-sse)
does not close this gap for the configured paper Trading API client: the stream
requires Broker API credentials. Its `FEE.details.parent_id` is optional for
one-to-one charges, while period-wide regulatory and other volume-based fees
have no parent execution ID. Access to that separate API, even if later
obtained, would require a completeness and allocation contract before a
verified *total* fee could be attached to each fill. No Broker API access or
additional paid data was obtained for this review.

Future read-only activity captures now include a broker-observed account
boundary (cash, equity, currency and observation time). The accounting CLI can
compare two such boundaries with a complete, covering same-account activity
snapshot using `--opening-account`, `--closing-account` and `--equity-output`.
It calculates execution cash effects from broker quantity and price, and uses
only explicitly timed USD `net_amount` rows for other cash activity. A row with
only a booked date, including the paper `FEE` schema observed here, remains
unresolved even if its date appears outside the interval; the booked date is
not an effective instant. A cash difference of even one cent is reported as
unverified. The output also reports the broker equity boundary change and
position-value residual, but never labels them verified strategy return or
allocates account charges to executions. The account observation precedes the
activity pagination, so it is not an atomic broker ledger snapshot.
The audit now labels observed cash distributions and position-changing
corporate actions separately. A precisely timed stock split with zero cash can
pass cash reconciliation while its share-count effect remains unverified;
date-only actions remain unplaced in the interval. Alpaca's
[Trading API activity schema](https://docs.alpaca.markets/us/docs/account-activities)
lists splits and dividends, but its nontrade object commonly supplies a date
rather than a causal execution instant. Neither the cash audit nor that schema
provides two position boundaries or a verified per-fill fee total.

The September 22 capture predates these account boundaries. No historical
cash/equity reconciliation is claimed from the seven fills or the 73 date-only
fee rows. A future interval needs two complete, same-account USD boundaries,
covering activity pagination, effective timestamps and cash amounts for all
relevant non-fill activity, plus position and per-execution fee evidence for
verified net strategy reporting.

## Historical ledger boundary

The original `trade_history.parquet` is preserved. The broker-backed rebuild
starts at the earliest available complete same-account paper position anchor,
September 7 at 03:19:26 (flat), and its 50 broker executions match the
September 18 20:01:12 closing quantity. The September 18 single-session ledger
also matches eight broker executions from a flat 13:29:43 opening. The
September 22 in-memory rebuild independently matched eight broker executions
from the flat 13:29:31 opening to the flat 20:01:53 closing. These are quantity
audits, not certification of fee completeness, execution-feed completeness,
tax lots, or strategy profit.

The existing bounded legacy diagnosis finds AAPL −3 and AMZN +1 **already at**
the verified flat September 18 opening. The 213 legacy rows timestamped before
the September 7 anchor have no account ID. The reviewed broker boundary and
activity inputs do not establish a same-account opening or complete execution
interval for those older rows. The precise first divergent event is unresolved. Do
not create an opening balance or edit historical records.

**Exclusion rule:** exclude pre-September 7 03:19:26 legacy history from the
broker-backed rebuild. Report the September 7–18 and September 22 verified
quantity intervals separately; do not join them through unverified days or use
legacy reconstructed AAPL/AMZN inventory as an opening balance. Historical
reconciliation requires an earlier complete same-account broker position
boundary plus execution-complete broker activities covering the disputed
interval, with identities that explain the two symbol differences. Until then,
the old interval remains explicitly unresolved.

## Research decision

**Retire the registered September 21 replacement-model hypothesis.** Its one
allowed trial is consumed and marked `hypothesis_rejected`. Five purged
development folds selected 30,802 proxy trades from 94,200 common
opportunities; zero folds were profitable. Mean net result was −3.197464 bps
per common opportunity, with a −3.445699 bps whole-session bootstrap lower
bound. The 10 bps cost assumption is a development proxy, not live execution
proof. No serving artifact or registry promotion resulted; the holdout was not
evaluated.

The existing replay qualification separately had 191 candidate samples against
250 required and about −0.129341 bps candidate net edge. It failed both the
minimum-sample and positive-net-edge checks. More samples alone do not repair
negative economics. Keep `required_model_stale`, replay control, and all cost
and promotion gates in force. Collect routine operational evidence only where
it closes the accounting gaps above; do not treat it as a new strategy trial.
A genuinely different research hypothesis needs a separate registered
specification and approval after the reset review, respecting the consumed
budgets and September 9–December 8 holdout. No experiment is proposed or run
in this task.
