# Small-account live risk contract — selected limits, evidence gate pending

This contract defines the evidence and calculations required before any new
live exposure. It is **not activated**. On September 23 the owner delegated
selection of the earlier rough 3% estimate. The selected policy is a 3%
peak-to-trough, cash-flow-adjusted account-equity drawdown ceiling and a
separate 1% daily account-equity loss ceiling. These are limits on new exposure,
not a guaranteed maximum realized loss or permission to trade live. At an
initial verified $1,000 account balance, the reference amounts are $30 and
$10; at $2,000 they are $60 and $20. Actual dollar limits must be computed from
verified same-account baselines, not the paper account or these examples.
The current paper profile and all model/replay/promotion gates remain in force.

## One account and one capital base

All risk inputs must identify the same broker account and be recorded with
broker/source timestamps and local receipt times. Use current broker equity as
the account's net liquidation value, with a separately recorded session-start
equity, starting-capital reference, and high-water equity. Cash deposits,
withdrawals, corporate actions and fees must be identified before comparing
equity across time; unknown cash activity makes the affected loss metric
unverified. Never substitute the paper account balance for the intended
$1,000–$2,000 live capital.

For an opening order, require a fresh same-account snapshot of equity, cash,
positions, open orders, fills/cancellations and unresolved submit intents.
Inventory gross and per-symbol exposure must include positions plus worst-case
remaining quantities of open orders. Pending buys reserve cash and costs;
pending sells do not release cash or exposure until a terminal broker result.
An accepted-but-unacknowledged order reserves its full requested quantity
until identity and cumulative fill state are reconciled. Missing quantities,
prices, sides, account IDs, timestamps or broker availability block new risk.

## Selected loss definitions and limits

- **Daily loss (1% ceiling):** adjusted equity at the start of the broker trading day minus
  current adjusted equity, floored at zero. This includes realized and
  unrealized P&L and actual posted fees; cash movements must be neutralized
  from the comparison. An estimated cost is disclosed separately and may be
  reserved conservatively, but cannot become a verified fee. Divide by the
  verified session-start adjusted equity, fixed for that trading day; block
  new exposure at or above 1% or when the denominator is unavailable.
- **Loss from starting capital:** approved starting-capital reference plus
  subsequent external cash flows minus current adjusted equity, floored at
  zero. A missing opening reference blocks this measurement; it must not be
  invented from later equity.
- **Peak-to-trough drawdown (3% ceiling):** highest verified cash-flow-adjusted equity since
  the approved start, minus current adjusted equity, divided by that peak.
  The high-water mark must persist across restarts and cannot reset at session
  rollover. Block new exposure at or above 3%. A corrupted or missing
  high-water record blocks new risk.

The owner has delegated the numeric choice above. The final enforcement must
use the more restrictive of the daily and drawdown limits, apply them to
realized and unrealized equity changes, and reserve capacity for outstanding
orders and conservatively estimated execution costs. If the potential loss of
an opening order cannot be bounded with verified exposure, it cannot consume
the remaining budget safely and must be denied. A software threshold cannot
guarantee realized loss when markets gap or broker execution is delayed. The
existing `live_canary` profile's static $25 daily-loss default is not the
selected 1% account-sized limit; it grants no authority to open live positions.
These percentages are conservative operator choices within the owner's rough
risk tolerance, not parameters optimized on paper trades or evidence of profit.
[FINRA's stop-order guidance](https://www.finra.org/investors/insights/stop-orders-factors-consider-during-volatile-markets)
notes that a triggered stop can execute far from its stop price during volatile
markets. The account loss ceilings are opening gates, not price guarantees.

## Authorization and safe reduction

New exposure is denied if any required snapshot is stale, incomplete or from a
different account; an unresolved submit or broker mismatch exists; any loss,
drawdown, gross/symbol exposure, cash or order-count limit would be breached;
or any existing model/replay/cost/freshness/provenance/promotion gate fails.
Order-count state must be durable and counted before broker submission.

Cancellation and a broker-verified order that only reduces an existing
position may proceed through its dedicated path, even while openings are
blocked, provided its side and quantity cannot flip the position and it does
not rely on stale or unknown exposure. Unknown exposure allows cancellation
and read-only reconciliation, not a guessed flatten order. Every direct broker
submission surface must either enforce this opening contract or prove its
risk-reducing status from fresh broker state.

## Acceptance before live capital

The canonical live opening precheck now refreshes the broker position and
open-order snapshot and account observation for each order, replaces any
caller-supplied exposure arrays, and rejects stale/failed broker reads. The
launch-profile evaluator rejects missing account identity, missing position or
open-order snapshots, unknown position market values, unpriced pending orders
and account conflicts. A closing label alone cannot prove a reduction whose
quantity would flip the position. These checks improve the existing live
opening path. The pure daily-loss evaluator still accepts a caller-supplied
`daily_loss_state` value without source identity, observation time, cash-flow
adjustment or durable session baseline. The canonical live opening precheck now
discards that value, so live openings fail closed at the daily-loss gate until
a verified source is implemented. A future implementation must derive loss
from fresh same-account broker equity and reconciled cash/activity evidence,
and block when any input is unknown. These checks do not establish a durable
high-water baseline,
enforcement of the selected loss thresholds, or proof that every direct broker submission and
live reduction path enforces this contract. The separate live-owner fence and
model/replay gates still apply.

Alpaca [defines `last_equity`](https://docs.alpaca.markets/us/docs/account-plans)
as equity at the prior trading day's 16:00 ET close, but
[nontrade activity records](https://docs.alpaca.markets/us/docs/account-activities)
can expose only a date rather than an exact instant. The
[activity endpoint](https://docs.alpaca.markets/us/reference/getaccountactivities-2)
also filters by creation time, while fees may be created the following day.
Consequently, a missing or date-only activity cannot certify that no intraday
cash movement affected the equity comparison. Live loss evidence must remain
blocked when that timing or completeness cannot be verified.

The selected numeric limits must be implemented in a single canonical evaluator
wired to every opening submission path, with same-account, timestamped broker evidence for
all inputs; durable high-water and session baselines; corruption, rollover,
restart, concurrent-submit, stale-state and ambiguous-order tests; an isolated
broker fault/restore rehearsal; passing code and replay gates; after-close
deployment with zero-position/open-order review; and a separately approved
live canary. Until each item is proven, the live-capital state remains blocked.
