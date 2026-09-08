# September 8 research checkpoint

Reviewed at 02:44 UTC on September 8, 2026. This records an evidence review and
a prospective development-study protocol; the new study has not been executed.

## Scheduled workflow result

The September 7 run completed at 21:16 UTC. The 23:00 verification reports
`workflow_executed` and `operator_report_verified`. Training used verified data;
the registry evaluated 12 named candidates. Accounting and paper review both
executed successfully. Registry selection remains blocked by unqualified
development evidence, insufficient samples, failed replay evidence and missing
shadow evidence. Execution success does not establish strategy eligibility.

Source: `/var/lib/ai-trading-bot/runtime/research_reports/evidence_verification/latest.json`,
generated September 7 at 23:00:56 UTC. Underlying run:
`/var/lib/ai-trading-bot/runtime/research_reports/daily/20260907T203713Z_daily/`.

Remaining gaps are `paper_session_not_reconciled`,
`execution_comparison_not_supported`, and `net_cost_validation_unavailable`.
The review has zero paired execution-cost orders; account-level fee activities
do not establish per-fill total fees. September 8 opening capture is expected
13:15-13:30 UTC and closing capture 20:00-20:15 UTC. Review after close, with the
existing verification timer scheduled for 23:00 UTC. A complete zero-trade
session may establish capture completeness but cannot validate execution costs.
Do not force trades or relax eligibility to collect samples.

## Fixed protocol: momentum at a longer horizon

Study ID: `fixed_momentum_30m_development_v1`. Status: protocol recorded,
evaluation pending. This is one exploratory study, not a promotion test.

Motivation: the existing one-bar controls have fold break-even round-trip costs
between -0.2013 and 0.3157 bps, far below the saved 6 bps assumption. Hypothesis:
holding a fixed positive-SMA-spread signal for 30 minutes produces enough gross
movement to exceed that cost with stable development performance. This is an
untested hypothesis, not evidence that longer holding improves returns.

- Freeze the governed dataset hash and five development windows from
  `artifacts/baseline_benchmark/protocol.json` and `baseline_benchmark.json`.
  Require the same acquisition provenance and quality gates. Do not consume
  the reserved holdout. These development windows have already informed the
  hypothesis, so results are exploratory and need subsequent untouched evidence.
- Use the canonical causal SMA-spread feature, computed using bars available
  through signal time. Signal is strictly `sma_spread > 0`; no fitted model,
  threshold search, symbol selection, feature search or alternate horizon.
- Evaluate AAPL, AMZN and MSFT. Signal at a completed minute bar, enter at the
  next minute bar open and exit at the open 30 minutes after entry. Require
  contiguous valid bars and both endpoints within the same regular session and
  development fold. Count every excluded opportunity by reason. Historical
  bar prices are research references, not observed executable quotes.
- Sample the common grid at 30-minute intervals anchored to the regular-session
  open, with the first signal after 30 completed bars. This prevents overlapping
  positions per symbol. Evaluate cash and always-long on the identical eligible
  grid and entry/exit prices. Do not treat cross-symbol opportunities as an
  account equity curve. Any future fitting must purge the full label interval
  at fold boundaries; no fitting is part of this study.
- Charge 6 bps round trip in the primary result; report fixed 0, 3, 6 and 10 bps
  sensitivities and gross break-even costs. Do not select a cheaper scenario to
  pass. Quote/fill-derived validation remains a separate evidence requirement.
- Report per-fold and per-symbol selected counts, common opportunity counts,
  gross and net edge per selection and per common opportunity, and paired net
  differences against both controls. Report a 95% session-block bootstrap
  interval for aggregate mean net edge per common opportunity, resampling whole
  sessions with all symbols together, 10,000 replicates and fixed seed 20260908.
- Continue only if there are five supported folds (at least 30 selections each),
  aggregate net edge and its bootstrap lower bound are positive at 6 bps,
  at least three folds are profitable, the aggregate paired difference against
  always-long is positive, and at least two symbols have positive net edge.
  These thresholds are research screening rules, not a profitability guarantee.
- Execute once. A valid failed result retires this exact protocol; do not tune
  it on the same windows. Missing or invalid data is inconclusive, with explicit
  rejection counts. Correct implementation errors only with recorded reasons
  and preserved prior artifacts. Passing permits planning an untouched test;
  it does not authorize deployment, portfolio integration or candidate revival.

## Validation

This change records reviewed artifacts and a protocol only. No runtime code,
model, trading setting or timer changed. No new behavior needs regression tests
or runtime smoke checks. Run documentation validation and `git diff --check`.
