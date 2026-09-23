# Small-account feasibility boundary

`ai_trading.tools.small_account_capacity` evaluates the same hypothetical
opening buy at $1,000 and $2,000 starting capital. It is a pure, long-only,
whole-share calculation. Callers must supply price, desired shares, existing
position market values, pending buy commitments, cash reserve, minimum and
maximum order notional, gross and symbol concentration limits, and per-side
spread/slippage plus an estimated total fee. No limit in this tool is an
approved live setting. Pending sell orders do not release capital until filled.

The result reports whole shares affordable under cash, gross exposure, symbol
exposure and order-notional limits; projected concentration; pending capital
commitments; and binding reasons. Estimated market cost and estimated fee are
separate from verified fees. When the fee assumption is unavailable, estimated
total cost is unknown. Verified total fee remains unknown in every scenario
because a synthetic sizing calculation cannot verify a broker charge.

This calculation assumes starting capital is unlevered equity and deducts
existing long positions and pending buys to estimate uninvested cash. It does
not reconstruct actual broker cash, borrow, tax lots, corporate actions, price
impact or queue priority. It cannot certify net performance or live readiness.
Synthetic tests cover both account sizes, whole-share/minimum-order failures,
pending capital and concentration, invalid inputs, and cost provenance. No
historical strategy trial or protected holdout was run.

## Decision rules for a future review

- **Retire:** a pre-registered hypothesis has consumed its permitted budget
  and has negative cost-adjusted development economics, as the September 21
  replacement-model trial did. Do not spend more samples to reverse the
  registered stopping decision.
- **Continue operational observation:** data needed to measure executions,
  fees, cash/equity reconciliation or broker state remains incomplete. Record
  the gap; do not treat abstention or paper fills as evidence of a positive
  trading edge.
- **Eligible to propose a live canary:** a separately approved hypothesis has
  positive development economics after explicit costs, all model/replay/
  freshness/provenance/promotion gates pass, verified account-size sizing and
  risk controls are enforced, execution fees and equity can be audited, and
  unattended recovery and monitoring have been demonstrated. Eligibility is
  not authorization to trade live. The September 9–December 8 holdout and
  consumed experiment budgets remain protected.
