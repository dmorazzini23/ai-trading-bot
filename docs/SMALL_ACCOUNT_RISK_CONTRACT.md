# Small-account live risk contract — pending owner limits

This contract defines the evidence and calculations required before any new
live exposure. It is **not activated**. The owner has not yet defined whether
the stated 3% maximum drawdown means a daily limit, loss from starting capital,
or peak-to-trough equity loss, nor supplied a separate daily-loss limit if
needed. No numeric loss or drawdown threshold is approved by this document.
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

## Loss definitions to be approved

- **Daily loss:** adjusted equity at the start of the broker trading day minus
  current adjusted equity, floored at zero. This includes realized and
  unrealized P&L and actual posted fees; cash movements must be neutralized
  from the comparison. An estimated cost is disclosed separately and may be
  reserved conservatively, but cannot become a verified fee.
- **Loss from starting capital:** approved starting-capital reference plus
  subsequent external cash flows minus current adjusted equity, floored at
  zero. A missing opening reference blocks this measurement; it must not be
  invented from later equity.
- **Peak-to-trough drawdown:** highest verified cash-flow-adjusted equity since
  the approved start, minus current adjusted equity, divided by that peak.
  The high-water mark must persist across restarts and cannot reset at session
  rollover. A corrupted or missing high-water record blocks new risk.

The owner must choose the 3% meaning and approve a separate daily-loss bound
if 3% is not daily. Limits must be expressed in both percentage and dollar
terms for the actual account capital. The final enforcement must use the more
restrictive of applicable limits and include a buffer for outstanding orders
and unknown execution costs. A software threshold cannot guarantee realized
loss when markets gap or broker execution is delayed.

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
opening path. Its daily-loss check currently accepts a caller-supplied
`daily_loss_state` value without source identity, observation time, cash-flow
adjustment or durable session baseline. That value is insufficient to authorize
live risk; a future implementation must derive the loss from fresh same-account
broker equity and reconciled cash/activity evidence, and block when any input
is unknown. These checks do not establish a durable high-water baseline,
approved loss thresholds, or proof that every direct broker submission and
live reduction path enforces this contract. The separate live-owner fence and
model/replay gates still apply.

Approval of the two numeric loss decisions; a single canonical evaluator wired
to every opening submission path; same-account, timestamped broker evidence for
all inputs; durable high-water and session baselines; corruption, rollover,
restart, concurrent-submit, stale-state and ambiguous-order tests; an isolated
broker fault/restore rehearsal; passing code and replay gates; after-close
deployment with zero-position/open-order review; and a separately approved
live canary. Until each item is proven, the live-capital state remains blocked.
