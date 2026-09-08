# Fixed baseline benchmark

Completed September 7, 2026, before tonight's scheduled research run.

The standalone `ai_trading.tools.baseline_benchmark` CLI evaluates cash,
always-long, and positive-SMA-spread momentum on the saved candidate's five
development outer-test windows. It records its protocol before evaluating,
requires the governed acquisition and matching dataset hash, verifies data
representativeness, rejects overridden label/cost contracts, and requires saved
chronological separation and label purging. Reconstructed test row counts must
match exactly. No model is fitted and no holdout is evaluated.

The hypothesis and stopping rule are fixed: compare the three controls once;
do not tune thresholds or revive the retired candidate. A non-cash baseline
qualifies only for further research if net edge per opportunity is positive
across at least five folds and at least 60% of folds are profitable. Cash is a
reference, not a profitable strategy.

## Actual results

The benchmark used 85,911 common opportunities from March 27 through July 15,
2026, for AAPL, AMZN and MSFT. The saved assumptions were 1 bps fees and 2 bps
slippage per leg, producing 6 bps round-trip costs on these inputs.

| Baseline | Selected opportunities | Net bps per common opportunity | Profitable folds |
| --- | ---: | ---: | ---: |
| Cash | 0 | 0.0000 | 0/5 |
| Always-long | 85,911 | -5.9260 | 0/5 |
| Momentum | 45,605 | -3.1042 | 0/5 |

Both trading baselines fail the criterion. The original candidate still selected
zero opportunities and remains retired on this evidence. These overlapping
equal-notional opportunity markouts are not equity returns and must not become
a portfolio-return ledger. Portfolio integration is not justified by this run.

Stress results at 0, 3, 6 and 10 bps round-trip cost and each fold's break-even
cost are included in the JSON. These are fixed sensitivity checks, not a search
for a cost assumption that makes a strategy pass.

## Artifacts and scheduled checks

- `artifacts/baseline_benchmark/protocol.json`: pre-evaluation protocol, source
  hash, saved windows and cost contract.
- `baseline_benchmark.json` in the same directory: paired comparisons, stress
  scenarios and decisions.
- `pipeline_with_baselines.json`: a derived copy containing the new comparisons;
  the original training and retirement artifacts are unchanged.
- `artifacts/research_decisions/research_decision_dashboard.html`: refreshed
  dashboard with all three baseline panels paired.

At the time of this review, the latest evidence verification still awaited the
September 7 run and September 8 session. Timers were confirmed active: daily
research next at September 7 20:37 UTC and read-only verification at 23:00 UTC
on September 7 and 8. Future results are not claimed complete.

## Validation and rollback

Tests exercise identical fold coverage, exclusion of a high-return observation
outside the development windows, baseline masks and costs, overlap rejection,
and nonfinite input rejection. The two focused tests passed. Required changed-file
validation passed lint, type, compile and 738 regression tests using
`bash scripts/agent_validate_changed.sh --skip-runtime-smoke`; `git diff --check`
also passed. Logs are in `/tmp/baseline-validation.log` and
`/tmp/baseline-tests.log`. This is an offline benchmark; no trading service
restart or decision change is required. Revert the new benchmark module/tests
to roll back, preserving the generated evidence and original candidate records.
