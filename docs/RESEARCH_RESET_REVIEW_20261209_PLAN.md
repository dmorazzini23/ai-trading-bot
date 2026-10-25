# December 9, 2026 review: pre-registered decision plan

**Recorded:** October 25, 2026, at the owner's request, before any holdout data
or momentum-trial outcome has been viewed. **Status:** adopted. It changes no
runtime behavior, gate, budget or authority. It fixes in advance *how* the
December 9 review will decide, so the result cannot steer the rules.

Amendments are allowed only as a dated record appended below, made before the
outcome they concern has been viewed. Any change made after viewing an outcome
voids that branch of this plan; the outcome is then treated as development
evidence only.

## Context at registration

- Week of October 19–23 (paper, diagnostics only): 13 one-share round trips in
  AAPL, AMZN and MSFT, about −$6.49 realized. 3 winners. Fills were at or better
  than the decision price; the losses came from selection, not execution. AAPL
  lost on all five days with the same pattern: an entry around 10:00 ET, exited
  about 20 minutes later lower.
- The day-sleeve model is 94+ days old against a 14-day limit and has run as
  shadow-only (`serving_decision=abstain`) since September. The replay/live
  parity gate fails at about −8 bps net.
- One registered trial is open: `cross_sectional_momentum_trial_20261018`
  ([amendment](RESEARCH_RESET_AMENDMENT_20261018_MOMENTUM.md), contract
  `config/cross_sectional_momentum_trial_campaign.json`, hash `5f694a82…a3e`).
  It is registered and not claimed; neither the support audit nor the preflight
  has run.

## Inputs the review will use (and only these)

1. The momentum trial ledger and report, verified against the contract hash.
2. The weekly research reset scorecards from October 31 through December 5,
   including the new reported-only **SPY benchmark** section (see below).
3. The status of the three named evidence gaps of the October 14 review (stale
   decision quotes, incomplete causal execution chains, per-fill total fees).
4. The replay/live parity gate and model registry state on December 9.

Paper P&L of the current intraday bot is reported but is **not** evidence for
or against any strategy: it runs a stale, abstaining model on one-share orders.

## Before December 9 (owner's word required for each step)

1. Run the timestamp-only support audit and the non-claiming `--preflight` for
   the momentum contract. Neither consumes the trial.
2. If both pass, claim the trial with `--fit` once, from a closed market
   window. It reads only 2024–2025 development data, so it can run before the
   review without touching the holdout.

If the trial has not been claimed by December 9, the review decides only
whether to claim it; it opens no holdout data.

## Decision rules

### A. Momentum trial

The contract's own acceptance rules apply unchanged (positive 95% bootstrap
lower bound vs cash at 10 bps, positive paired lower bound vs equal-weight
always-long, positive aggregate at 20 bps, at least 5 of 8 profitable
quarters).

| Trial outcome | Decision |
| --- | --- |
| `eligible_for_untouched_test_planning` | Run **one** untouched evaluation (A1). |
| `retired` or `inconclusive_budget_consumed` | Retire the hypothesis. No tuning of lookback, quantile, hold, universe, cost or completeness. Go to C. |

**A1. Untouched evaluation.** The frozen contract, unchanged, is scored once on
the six-ETF universe over September 9–December 8, 2026 (about 12
non-overlapping 5-session periods). Twelve periods cannot give a meaningful
bootstrap bound, so this is a consistency check, not proof:

- **Pass:** mean net return per period at 10 bps > 0 **and** mean paired
  difference vs equal-weight always-long ≥ 0.
- **Fail:** anything else, including missing data. The hypothesis is retired.

The holdout is then spent for this hypothesis: it is never re-scored, and a
fail is never followed by a variant on the same window.

**A2. After a pass.** Draft a registered paper-trading amendment for the
momentum strategy as a separate sleeve (6 ETFs, 5-session holds, its own risk
caps). Wiring it into the runtime needs its own amendment and the existing
validation; it is not automatic.

### B. Current intraday day-sleeve model

- If no qualified candidate exists on December 9 (no model passing the
  existing promotion and parity gates), **retire the day-sleeve model from
  runtime** rather than retraining it. Removing it clears the
  `required_model_stale` flag honestly instead of refreshing a model that has
  not shown an edge.
- The AAPL morning-entry pattern above is a paper diagnostic observed inside
  the holdout period. It may become a hypothesis only in a new registered
  contract tested on data outside September 9–December 8. It is not a reason
  to change the live bot before or at the review.

### C. If nothing passes

Continue the pause and paper operations. A new campaign needs a pre-registered
contract whose timestamp-only sampling-support audit passes before a trial is
claimed (October 14 rule). The holdout window stays reserved for any hypothesis
not yet scored on it.

## Real-money gate (fixed now)

A small live canary may be proposed only when **all** of the following hold:

1. A strategy passed A1 and then ran as a paper sleeve for at least **8 weekly
   scorecards** and at least **30 closed positions**.
2. Its paper P&L is positive **after measured costs** (broker fees where
   available, otherwise the contract's 10 bps tier; never assumed zero).
3. Over the same weeks, it is at least even with the **exposure-matched SPY
   benchmark** (SPY held over the same intervals with the same notional).
4. `live_capital_readiness` reports `live_canary_allowed`, and the canary uses
   the `live_canary` profile ($25 daily loss cap, no shorts).
5. The owner approves explicitly. Nothing in automation changes live authority.

## SPY benchmark in the weekly scorecard

From October 25 the weekly scorecard reports two comparisons, both reported
only (they never become blockers or gates):

- **Account vs SPY:** the paper account's equity return over the week's
  sessions next to SPY's close-to-close return over the same sessions. This
  answers "would the money have done better sitting in SPY?". While the bot
  trades one share, the account is mostly cash, so this mainly shows the cost
  of being idle.
- **Trades vs exposure-matched SPY:** for each closed same-day round trip, the
  P&L SPY would have made with the same notional, held from the same entry
  minute to the same exit minute. This answers "did the trade picks beat just
  being long the market while in a trade?".

## Change record

- Added this plan and linked it from `docs/RESEARCH_RESET.md`.
- `config/research_reset.json` is unchanged; `review_date` stays `2026-12-09`.
- Rollback: revert the commit. No runtime state depends on this document.

## Amendment record

- **2026-10-25 — momentum trial outcome (no rule change).** The trial was
  claimed once and ended `retired` on all four acceptance criteria
  ([result](RESEARCH_RESET_AMENDMENT_20261018_MOMENTUM.md#result-october-25-2026-retired)).
  Branch A ends; A1 does not run and the holdout stays reserved. The review
  proceeds under B and C.
- **2026-10-25 — volatility-target trial outcome (no rule change).**
  `volatility_target_core_trial_20261025` was registered, acquired, preflighted
  and claimed once; it ended `retired` because the Sharpe-difference lower
  bound was not positive (CI −0.14..+0.26). The drawdown, stress-cost and
  turnover criteria passed
  ([result](RESEARCH_RESET_AMENDMENT_20261025_VOLATILITY_TARGET.md#result-october-25-2026-retired)).
  The holdout stays reserved. The review proceeds under B and C.
