# Proposal: portfolio manager mode (for the December 9 review)

**Date:** 2026-10-25
**Status:** draft for the owner to decide at the
[December 9 review](RESEARCH_RESET_REVIEW_20261209_PLAN.md). Nothing is built,
configured or deployed. The live bot keeps paper trading unchanged until then.

## 1. Why

Six registered trials have now ended the same way. None beat holding the
market with enough confidence to risk money:

- liquidity reversal: inconclusive;
- 60-session ETF trend: inconclusive;
- logistic day model: retired;
- day-sleeve horizon: inconclusive;
- cross-sectional momentum: retired;
- volatility-targeted core: retired.

The volatility trial is the clearest example. It cut the worst drawdown from
30% to 19%, but it also cut return from 15.7% to 13.3% a year, and the gain in
risk-adjusted return could not be told apart from zero.

The honest conclusion is to stop asking the bot to find an edge. Instead, ask it
to **run a plain, owner-chosen investment plan reliably**:

- hold a fixed diversified mix;
- rebalance it on rules;
- enforce safety limits;
- report honestly against benchmarks.

This mode claims **no edge**. Its expected return is the market's return on the
chosen mix, minus small costs. Its value is discipline, automation and honest
reporting.

## 2. What the mode does

1. **Holds a fixed target mix** of a few broad, liquid, low-cost ETFs from
   `config/portfolio_plan.json`. The owner chooses it (section 6).
2. **Checks daily, trades rarely.** Once per session, after 10:00 ET, it
   compares current weights with the targets. It trades only when:
   - an asset is outside its **tolerance band** (for example ±5 percentage
     points absolute), or
   - the scheduled **calendar rebalance** comes due (for example quarterly),
     and drift exceeds a small minimum.
3. **Uses cash flows first.** New deposits buy the most underweight assets, and
   withdrawals sell the most overweight. This avoids selling just to rebalance.
4. **Places careful orders:**
   - regular hours only, marketable limit orders with the existing spread and
     quote-age gates;
   - idempotent client order IDs per rebalance, so a retry can never double an
     order;
   - sells before buys;
   - whole shares, with the leftover in cash.
   - Fractional or notional orders need to be verified on the current
     `alpaca-py` order path before they are relied on.
5. **Never times the market.** No signals, models, momentum or volatility
   rules. All of those were tested and retired.

## 3. Safety rules

The risk framework changes from limiting daily trading losses to protecting a
long-term holding. Some of the current day-trading controls would do harm here.

| Control | Day-trading bot today | Portfolio manager mode |
| --- | --- | --- |
| End-of-day flatten | On (`AI_TRADING_EOD_FLATTEN_ENABLED=1`) | **Off.** Holdings are meant to stay overnight. |
| Daily loss cap | Stops trading after a loss, e.g. $25 in `live_canary` | Replaced. A market drop is not a trading error, and forced selling at lows locks in losses. Drawdowns raise **alerts** (e.g. −10%, −20%), never sales. |
| Kill switch | Halts trading | Kept. **Halts new orders only**; it never liquidates. |
| Allowed symbols | Day-trading list | Only the ETFs in the plan, plus the cash ETF. |
| Leverage, shorts, margin | No shorts in paper | **None.** Cash-only account; the startup check that the broker account settings match (`no_shorting`, multiplier 1) stays required. |
| Order limits | Per-order notional, max orders per day | Kept. Sized to the plan: max orders per rebalance, max notional per order, and max total traded per day as a share of the portfolio. |
| Manual approval | `manual_approval_required` in live profiles | Kept for live. Each rebalance is proposed (what, why, cost estimate) and placed only after the owner approves; it could later relax to automatic within the bands. |
| Data and quote gates | Alpaca-only quotes, spread and age caps | Kept unchanged. |
| Model, replay and promotion gates | Block trading (stale model) | Not applicable: no model. The mode refuses to start if any signal or model path is enabled. |

## 4. What gets switched off and what is kept

**Switched off in this mode:**
- the intraday ML pipeline, the day sleeve, the netting signal cycle and paper
  sampling;
- end-of-day flatten;
- scheduled training, which is already paused.

**Kept:**
- the service, health endpoints and identity checks;
- the OMS ledger and invariants, TCA records and the trade-updates stream;
- the broker account configuration check;
- the Slack preflight and close reports;
- the weekly scorecard and the SPY benchmark;
- runtime backups.

## 5. Reporting

**Weekly Slack summary:**
- current versus target weights and drift;
- trades placed and why (band or calendar);
- measured costs (spread, slippage and fees where available);
- portfolio return next to two benchmarks:
  1. **The plan benchmark:** the same target mix rebalanced perfectly at zero
     cost. The gap shows how well the bot executes the plan; it should be
     small and slightly negative, roughly the costs.
  2. **SPY:** context only. A mix with bonds will usually trail SPY in rising
     markets and fall less in sell-offs. That is the choice the mix makes, not
     a fault of the bot.

**Monthly:**
- a reconciliation of internal positions and P&L against the broker's
  portfolio history; the existing cross-check is extended;
- the worst drawdown to date.

## 6. Decisions for the owner at the December 9 review

These are personal investment choices, not things this project can test or
decide.

1. **The mix.** How much in stocks versus bonds or cash, set by your goals,
   time horizon and comfort with drops.
   - Illustration only, not a recommendation: a broad US stock ETF, an
     international stock ETF and a bond ETF.
   - Selection criteria: broad index, low expense ratio, high liquidity.
2. **Account type and tax treatment.** Rebalancing sells can create taxable
   gains in a taxable account. A tax-advantaged account, cash-flow
   rebalancing and wider bands reduce this. Consider professional tax advice.
3. **Amount and pace.** How much real money, whether to invest all at once or
   in steps, and the first live amount.
4. **Rebalancing rule.** Band width and calendar frequency.
5. **Approval mode.** Approve every rebalance (recommended at first), or
   automatic within bands.
6. **The intraday bot.** Retire it fully, or keep it in paper as a monitoring
   exercise. The recommendation is to retire it: it has no edge and runs a
   stale model.

## 7. Rollout and gates

The December 9 real-money gate was written for a strategy that **claims an
edge**: positive after costs and at least even with SPY. A plain mix claims
none, so that gate does not fit. The review should adopt an **operational
gate** instead, by amendment:

- **Phase 0, build (offline):**
  - plan config and rebalance logic, with unit tests for band and calendar
    triggers, cash-flow allocation, rounding, sells-before-buys, idempotency,
    and refusal of unknown symbols, leverage and shorts;
  - the launch profiles `portfolio_paper` and `portfolio_live`;
  - EOD flatten off and the signal path refused in this mode;
  - the weekly report;
  - full validation and CI; no live change.
- **Phase 1, paper:** at least **8 weekly scorecards** in paper with a
  paper-sized plan. Pass conditions:
  - every rebalance was correct (right trigger, right sizes);
  - zero unintended orders;
  - zero reconciliation breaks;
  - the kill switch and approval flow tested once each;
  - the gap to the plan benchmark is within measured costs.
- **Phase 2, small live:** a small owner-chosen amount, manual approval for
  every rebalance, and the same reports. At least 3 months.
- **Phase 3, normal live:** the owner decides on the full amount and the
  approval mode. No automation ever changes the amount, the mix or the live
  authority.

**Rollback at any phase:** set the profile back to `paper_observe`. That stops
all orders and leaves the holdings in place, in a plain ETF position the owner
can manage directly at the broker.

## 8. Rough scope

The engineering is small next to the research work already done.

- **New code:**
  - a plan config;
  - a rebalance module (roughly 300–400 lines) with tests;
  - two launch profiles;
  - a mode switch that disables the signal path and EOD flatten;
  - weekly report sections.
- **Reused:** order submission, idempotency, quote gates, OMS, TCA, health,
  Slack and backups.
- **Main risk:** keeping the day-trading code paths off in this mode. Each one
  needs a test proving it is inert.

## 9. What this mode is not

- It is not a source of extra return. Over long periods it should track the
  chosen mix minus a few basis points of cost.
- It is not protection from market drops. A stock-heavy mix will fall with the
  market; only the mix itself sets how much.
- It is not financial advice. The mix, the amount and the account are the
  owner's decisions.
