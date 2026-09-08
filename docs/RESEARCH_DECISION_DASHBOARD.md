# Research decisions and portfolio evaluation

The audit found existing chronological common-opportunity baseline comparisons,
fold opportunity diagnostics, experiment stopping rules, and capital-control
reports. The new dashboard combines those sources; it does not replace their
selection or promotion gates.

`ai_trading.tools.research_decision_dashboard` produces JSON and a searchable,
self-contained HTML report. The daily workflow runs it after paper evidence
review and exposes its HTML path in the operator summary. Inputs are hashed.
Unchanged accelerator results resolve their previous report before rendering.

The report covers four capabilities:

- Opportunity accounting: each fold reconciles opportunities into invalid
  scores, frozen-threshold rejections and selections. Session and replay
  diagnostics cover observed fills, rejection reasons and caps separately.
  These populations are not falsely joined into one execution funnel.
- Independent baselines: cash, always-long and momentum are paired against the
  candidate only on matching folds and opportunity counts. Existing training
  code computes these on development outer folds with nested threshold selection
  and common costs. Missing comparisons remain missing.
- Portfolio evaluation: fixed equal initial capital is split across independently
  compounded strategy sleeves, before and after adding a candidate. Cost scenarios
  use 0, 3, 6 and 10 bps per traded notional. Outputs include total return,
  drawdown, incremental daily return, loss coincidence, correlation, turnover,
  exposure and same-day exposure overlap. No allocation is optimized.
- Research decisions: development-only retirement diagnostics, experiment ledger
  state, paired baseline results and source provenance appear together.

## Portfolio input contract

The daily workflow optionally consumes
`runtime/research_portfolio_returns.csv` and
`runtime/research_portfolio_returns.manifest.json` under the configured runtime
root. Set `AI_TRADING_RESEARCH_PORTFOLIO_CANDIDATE` to the candidate strategy ID;
its default is `candidate`.

CSV columns are `session,strategy,gross_return,turnover,gross_exposure`.
Sessions are UTC dates; every strategy must explicitly cover every session,
including zero-return/no-trade dates. Returns are daily equity fractions before
the tested cost assumption. Turnover is traded notional divided by each sleeve's
NAV, including its entries, exits and internal rebalancing. Gross exposure is
between zero and one. Sleeves receive equal initial capital and do not rebalance
capital between each other. At least 20 aligned sessions are required for an
`evaluated` status. Missing/duplicate rows, nonfinite values, invalid exposures
and capital-exhausting scenarios are rejected.

The manifest must provide the CSV `sha256`,
`evidence_partition: out_of_sample`, `return_basis: daily_equity_fraction`, and
`turnover_basis: traded_notional_over_sleeve_nav`. These declarations bind the
input contract; they do not independently certify the upstream backtest.
Opportunity markouts must not be supplied as equity returns. Same-day overlap
does not establish intraday liquidity capacity or fill feasibility.

## Current evidence and validation

`artifacts/research_decisions/research_decision_dashboard.html` is the current
report. The inspected retired candidate has five reconciled folds and zero
selected opportunities. Its run disabled controlled baseline experiments, so
paired baseline results are missing. No aligned portfolio return ledger exists;
the portfolio panel remains pending. The retired candidate was not retrained and
no untouched holdout was consumed to fill these gaps.

Regression tests cover count reconciliation, baseline pairing, same-capital
portfolio compounding, costs, missing/duplicate/invalid returns, and HTML
escaping. The existing daily plan test verifies dashboard wiring. Validation is
recorded in `/tmp/decision-dashboard-validation.log` and focused checks in
`/tmp/decision-dashboard-tests.log`.
Changed-file validation passed 736 tests, lint, mypy and compilation. The final
dashboard and daily-workflow suite passed 28 tests. The non-sending incident
check passed; live health returned HTTP 503 for the existing stale-model gate
with broker connectivity healthy. Current evidence accounts for 85,911 scored
opportunities, all rejected by the frozen policy, and zero invalid scores.

Rollback: remove the dashboard step and operator-summary/latest-report wiring,
then revert the new module and tests. Research processes load the module on
startup; this change does not require restarting the trading service.
