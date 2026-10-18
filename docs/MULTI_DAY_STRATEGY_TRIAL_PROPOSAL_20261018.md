# Proposal: one registered multi-day strategy trial (cross-sectional momentum)

**Date:** 2026-10-18
**Status:** draft for the owner. Nothing is registered, claimed or fitted. No
strategy signal or return has been computed, and no price column has been read.
The holdout (2026-09-09 to 2026-12-08) and all 2026 data stay closed.

Contract draft: `config/cross_sectional_momentum_trial_campaign.json`
(`"status": "draft_not_registered"`). Runner: `ai_trading/tools/registered_strategy_trial.py`.
Support audit: `ai_trading/tools/research_feasibility.py` (daily rebalance grid).

## 1. Why this hypothesis

Three registered intraday trials failed after a fixed 10 bps round-trip cost:
the September 21 replacement model (`docs/MODEL_REPLACEMENT_20260921.md`), the
30-minute momentum study (`docs/MOMENTUM_30M_RESULTS.md`) and the 60-minute
day-sleeve horizon trial (`docs/DAY_SLEEVE_HORIZON_TRIAL_RESULT_20261018.md`).
In large caps, a gross move within a session is about the same size as the cost.
`docs/STRATEGY_WIRING_AUDIT_20261018.md` §5 recommends the cheapest multi-day test.
It runs the **unmodified** `CrossSectionalMomentumStrategy` offline through the
registered-trial machinery and does not touch the live pipeline.

**Hypothesis** (`cross_sectional_momentum_63_session_top_quintile_long_only_5_session_hold_v1`):
the registry class `cross_sectional_momentum` uses its constructor defaults
(63-session lookback, top quantile 0.2, `min_universe` 6), and only its `buy`
signals are scored. It earns a positive net return after a 10 bps round trip.
It also beats an equal-weight always-long basket of the same universe on the
same dates. It rebalances every 5 sessions, and holds never overlap.

The economic motivation is medium-horizon relative strength across sectors and
asset classes (Moskowitz and Grinblatt 1999; Asness, Moskowitz and Pedersen 2013).
A 5-session gross move is several times the cost. This tests an existing class
as written. It is not a new model, feature or parameter search.

## 2. Universe choice: support audit against both verified datasets

The universe was chosen by the timestamp-only daily support audit. It reads
only the `timestamp` column of hash-verified sources. No data were fetched.

Grid: the decision is made after the close of session *t*, using closes up to *t*.
Entry is at the open of *t+1* and exit at the open of *t+6*. Decisions run at
*t* = 63, 68, …, 493 of the 502 development sessions, so there are 87 candidate
periods. A period counts only if every symbol has a complete session for all 70
sessions from *t*−63 through *t*+6. Thresholds were fixed before the audit, from
the ETF trend precedent: ≥60 decision periods, ≥60 selections, and ≥6 quarters
(by entry session) with ≥10 selections each.

| Universe (acquisition) | Completeness rule | Possible periods | Buys per period | Possible selections | Quarters ≥10 | Status |
|---|---|---|---|---|---|---|
| AAPL, AMZN, MSFT (`artifacts/stock_development/acquisition.json`) | either | 87 | 0 (3 < `min_universe` 6) | 0 | 0 | **blocked** |
| DIA, IWM, QQQ, SPY, XLE, XLF (`artifacts/research_foundation/acquisition_sip_repaired.json`) | `every_minute` | 3 | 1 | 3 | 0 | blocked |
| same ETFs | `open_and_close_minutes` with 98% coverage floor | 20 | 1 | 20 | 1 | blocked |
| **same ETFs** | **`open_and_close_minutes`, no coverage floor (contract)** | **87** | **1** | **87** | **7** | **support possible** |

Selections per quarter under the contract rule: 2024Q2 12, 2024Q3 13, 2024Q4 13,
2025Q1 12, 2025Q2 12, 2025Q3 13, 2025Q4 12. The first entry is 2024-04-04 and the
last is 2025-12-19. Audit identity: protocol hash
`da11080021c17d286b95c736c50f932a643b8b4cb0c93561beb75309e35b1d7b`, acquisition
sha256 `f1acd5852fcae1be…`, manifest sha256 `aa27ebde808ea40c…`. Flipping `status`
to `registered` changes the protocol hash. The preflight re-runs this audit for
the registered contract.

**Stocks cannot be used.** With three names, `generate_signals` returns `[]` on
every call because the strategy requires six. Any trial on that set would be
inconclusive by construction.

**ETFs, and why this completeness rule.** DIA has 973 missing minutes across 243
sessions. None of them is an opening or closing minute, and the largest gap in a
single session is 16 minutes. The strategy uses only the session close (signal)
and the session open (entry and exit). On 2026-09-09, `docs/RESEARCH_RESET.md`
named this as the candidate data contract for a future campaign: "require only
the completed closes and executable entry/exit observations used by the fixed
rule". The contract adopts that rule. A session counts if its first and last
regular minutes are present and every present minute has valid price geometry
and positive volume. There is no interpolation and no invented bar.

Disclosure: I first audited a 98% minute-coverage floor (from timestamps only).
It blocked, because 27 DIA sessions fall below 98% minute coverage. I then
dropped the floor to match the reset's own wording. No price, signal or return
was seen at any step. The owner may reject this rule. If so, no supported
universe exists in the verified data, and the next step is a new acquisition
(wiring audit §5, item 5), not a trial.

**Horizon.** The 5-session hold follows wiring audit §5 item 6 and the ETF
precedent. Timestamp-only counts for the alternatives under the contract rule:
10 sessions gives 43 periods and 21 sessions gives 20. Both are below the
60-period floor.

## 3. Contract summary (all fixed before evaluation)

- **Strategy:** `REGISTRY["cross_sectional_momentum"]()` with no arguments. The
  runner refuses to run unless `lookback`, `top_quantile` and `min_universe`
  equal 63, 0.2 and 6, and the grid lookback equals the strategy lookback. Short
  signals are ignored.
- **Data:** 2024-01-01 to 2025-12-31, SIP, split-adjusted, regular-session daily
  OHLCV aggregated from verified minute bars. Early closes use their own session end.
- **Allocation:** equal notional across the period's buys. With six names that
  is one buy per period. Abstention is cash. No leverage, shorts or compounding.
- **Costs:** each selected slot pays a full round trip every period. The control
  pays the same, so the paired comparison with always-long is cost-neutral.

  | Tier | bps | Role |
  |---|---|---|
  | primary | 10 | decision |
  | stress_high | 20 | aggregate net must be positive |
  | stress_low | 6 | reported only |
  | measured_paper_taker_median | 2.8 | reported only; can never change the decision |

  The 2.8 bps tier is the median decision-time quoted spread from
  `docs/PAPER_ROUND_TRIP_COST_EVIDENCE_20261018.md` (p75 5.63, p95 25.3). It was
  measured on AAPL, AMZN and MSFT paper fills, not on ETFs. It is declared now so
  that it cannot later be used to rescue a failed result.
- **Controls:** cash, and an equal-weight always-long basket of all six ETFs on
  the same entry and exit opens.
- **Bootstrap:** circular blocks of 4 decision periods, 10,000 replicates, seed
  20261018, applied to the same resampled indices for both comparisons.
- **Acceptance (all required):** adequate support; a positive 95% lower bound of
  net per period versus cash at 10 bps; a positive lower bound of the paired
  difference versus always-long; positive aggregate net at 20 bps; at least 5
  profitable quarters at 10 bps.
- **Decisions:** all criteria met → `eligible_for_untouched_test_planning`;
  adequate support but any criterion failed → `retired`; inadequate support
  after the claim → `inconclusive_budget_consumed`.
- **Stop rule:** one trial. The claim is made immediately before signals are
  generated, and there is no retry after a claim, even after a crash. A failed
  preflight or support audit blocks without spending the trial. Failure retires
  the hypothesis and authorizes no tuning of lookback, quantile, hold, universe,
  cost or completeness rule.
- `promotion_authority: false`, `runtime_authority: false`.

## 4. Runner semantics (`ai_trading/tools/registered_strategy_trial.py`)

Fail-closed order, matching `model_replacement_trial`:

1. Refuse a contract whose `status` is not `registered`. Validate the strategy
   defaults, grid, cost tiers, thresholds and bootstrap. Refuse a development
   window that overlaps the holdout or ends in 2026. All of this happens before
   any file read.
2. Require `campaign_id` in `config/research_reset.json` `registered_trials` and
   an existing `artifacts/<campaign_id>/campaign_state.json`. The runner never
   creates a budget. `register_campaign` verifies the contract is unchanged. If
   any trial is already recorded, return `budget_already_claimed` without
   reading data.
3. `load_governed_development_timestamps`: acquisition identity, source sha256,
   stable reads, timestamps only. Refuse any row outside the development window,
   in the holdout or in 2026 **before** prices are read.
4. Run the daily support audit. If it blocks, write `preflight.json` and exit
   without a claim.
5. Read prices only from hash-verified sources whose timestamps equal the audited
   ones, build the daily bars, and recount support after price validity. Write
   `preflight.json` with the evidence signature (contract hash, provenance, code
   hashes). `--preflight` stops here.
6. `--fit`: `claim_campaign_trial`, once. At each usable decision, call
   `generate_signals({"prices": {sym: closes[0..t]}})`, keep the buys and score
   them. Validate the periods (entry = *t*+1, exit = entry+5, no overlap, inside
   development). Then evaluate, write `periods.csv` and `report.json`, and call
   `finish_campaign_trial`.

## 5. Exact commands for the owner

Run from `/home/aiuser/ai-trading-bot` as `aiuser`, after the market closes.

```bash
# 0. Adopt the amendment (section 6): add the campaign to config/research_reset.json
#    and set "status": "registered" in config/cross_sectional_momentum_trial_campaign.json.

# 1. Register (creates the one-trial budget; immutable afterwards).
./venv/bin/python -c "import json; from pathlib import Path; from ai_trading.tools.experiment_ledger import register_campaign; c = json.loads(Path('config/cross_sectional_momentum_trial_campaign.json').read_text()); print(register_campaign(Path('artifacts/cross_sectional_momentum_trial_20261018/campaign_state.json'), c)['contract_hash'])"

# 2. Timestamp-only support audit for the registered contract (no claim).
./venv/bin/python -m ai_trading.tools.research_feasibility \
  --protocol config/cross_sectional_momentum_trial_campaign.json \
  --sampling-acquisition artifacts/research_foundation/acquisition_sip_repaired.json \
  --output artifacts/cross_sectional_momentum_trial_20261018/support_audit.json

# 3. Non-claiming preflight (hashes, bounds, daily bars, price-validated support).
./venv/bin/python -m ai_trading.tools.registered_strategy_trial \
  --repository-root . --contract config/cross_sectional_momentum_trial_campaign.json --preflight

# 4. Only if step 3 reports preflight_passed: claim once and evaluate.
./venv/bin/python -m ai_trading.tools.registered_strategy_trial \
  --repository-root . --contract config/cross_sectional_momentum_trial_campaign.json --fit
```

## 6. Amendment text the owner would adopt

> # Research reset amendment: <date>
>
> **Decision:** the owner authorizes **one** registered development trial,
> `cross_sectional_momentum_trial_20261018`, under the existing
> `registered_development_trial` allowance of `config/research_reset.json`.
> The pause otherwise continues unchanged. The next explicit review remains
> **December 9, 2026**.
>
> **Scope.** The contract is frozen in
> `config/cross_sectional_momentum_trial_campaign.json` and registered before any
> outcome is viewed, in
> `artifacts/cross_sectional_momentum_trial_20261018/campaign_state.json`. The
> trial reads only the verified 2024–2025 SIP development data for DIA, IWM,
> QQQ, SPY, XLE and XLF (`artifacts/research_foundation/acquisition_sip_repaired.json`)
> under the `open_and_close_minutes` session rule. The September 9 to December 8
> holdout is not opened, and no 2026 data is read. The trial tests one
> hypothesis: the unmodified `CrossSectionalMomentumStrategy`, long-only, with a
> 5-session non-overlapping hold, earns positive net returns after a fixed 10 bps
> round trip and beats equal-weight always-long. The budget is one trial. A
> failing or inconclusive result retires the hypothesis and authorizes no tuning.
> The 2.8 bps measured-paper tier is reported only.
>
> **What a passing result permits.** Planning only: a separately registered
> evaluation on a broader, rule-selected universe and on untouched data. It does
> not wire the strategy into the live netting cycle, add a sleeve, change
> allocation weights, promote anything, relax any gate, or change paper or live
> trading authority.
>
> **Unchanged.** Scheduled training, new features, new model families and
> unregistered searches stay paused. Promotion, staleness and parity gates are
> unchanged. The consumed ETF trend, September 21 replacement and day-sleeve
> horizon budgets are unchanged. The October 14 evidence gaps remain the
> priority. This amendment neither resumes broad research nor moves the review date.
>
> **Preconditions before the claim.** The timestamp-only daily support audit
> passes for the registered contract. The non-claiming preflight passes. Only
> then may `--fit` claim the trial, once.
>
> **Change record.** `config/research_reset.json`: append the campaign to
> `registered_trials` and add an `amendments` entry. Contract `status` becomes
> `registered`. The ledger is created by `register_campaign`. Rollback: revert
> the config edits; keep any ledger as evidence.

## 7. Known limitations and risks

- **The data are not untouched.** The September ETF trend study used the same six
  ETFs and window. Its descriptive four-period result is in
  `docs/RESEARCH_RESET.md`. The new hypothesis is a different rule, but this is
  still a selection-bias limitation.
- **One selection per period.** The result is concentrated in one name at a time.
  The report records selections by symbol.
- **Price validity is unchecked so far.** The audit counts only timestamps. The
  preflight may still reject DIA sessions for invalid prices, and it blocks
  without a claim if support then falls below the thresholds.
- **Markouts, not total returns.** Split-adjusted open-to-open markouts omit
  distributions, cash interest, fills and fees.
- **Cost tier source.** The 2.8 bps tier comes from stock fills, not ETFs.
