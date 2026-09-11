# Adjusted-data preparation: stop before a new trial

Completed September 9, 2026 under the user's end-to-end authorization.
**Decision: data acquisition and observation feasibility pass; strategy evaluation
does not pass its prerequisites. Stop before registration.** No profitability
claim, new trial, model installation or order follows from this work.

The recommendation is to shelve this strategy direction until independent
execution-cost and prospective-evaluation evidence can support reopening it.
No further parameter searches or automatic retries are warranted by these results.

## Completed evidence

| Check | Result |
| --- | --- |
| Acquisition | 1,167,227 SIP one-minute bars, all-adjusted, six ETFs, 2024–2025 |
| Governed quality | All six pass; manifest and exact CSV hashes verified |
| Required observations | 88 shared valid periods; at most 528 selections |
| Corporate actions | 64 cash dividends and one XLE 2-for-1 split with development ex-dates |
| Predetermined action samples | First/middle/last dividend per ETF and every split: 19 samples |
| Adjustment diagnostic | All 19 consistent under the explicit cent-rounding assumption |
| Sample-to-dataset linkage | All 38 before/after all-adjusted closes match saved dataset bytes |
| Strategy outcomes | No signals, selections or returns calculated |

DIA still has 973 missing interior minutes. The exact required observations pass;
this does not make the full minute history complete. The acquisition retained the
existing 2% missing-minute gate; it was not loosened. Source pagination completed.

The corporate-action request was widened to December 2023–March 2026 because the
initial date range included actions with earlier ex-dates and later payable/process
dates. Only ex-dates in 2024–2025 enter this review. Provider pagination completed,
but a complete API response is not independent proof that every action exists.

The provider documents [price adjustment options](https://docs.alpaca.markets/us/reference/stockbars)
and [corporate-action completeness filtering](https://docs.alpaca.markets/us/reference/corporateactions-1).
These document semantics; they do not certify this dataset's execution realism.

## What remains unverified

The dividend diagnostic compares the ratio of all-adjusted to split-adjusted
closes across an ex-date with `1 - cash_distribution / previous_raw_close`.
Split diagnostics compare split/raw ratios with the reported old/new share ratio.
This cancels common later adjustment factors. Price inputs are matched in time.

All checks are consistent within an interval assuming each displayed price rounds
to a cent. That assumption was used for diagnosis after inspecting the samples;
it is **not a preregistered acceptance threshold**. Provider rounding precision and
the official reference close were not certified. The final regular minute close
is only a reference proxy. The report remains diagnostic, not full corporate-action
certification. Future price revisions and point-in-time signal construction also
require a defined policy before evaluation.

The 10/20/40 bps cost scenarios remain assumptions. Existing account evidence does
not verify complete per-fill fees or attainable opening execution costs. Today's
zero fills cannot close that gap. No fees were invented or allocated.

## Evaluation plan and stopping decision

`config/endpoint_evaluation_plan.json` specifies the hypothesis, equal-slot
allocation, paired cash/always-long controls, fixed costs, joint-period block
bootstrap, sample requirements and one-trial stopping rule. Its exact bytes are
frozen in `artifacts/endpoint_proposal/evaluation_plan_freeze.json`.

The September 9–December 8, 2026 holdout stays reserved and unread. The plan describes
a separate prospective observation interval of December 9, 2026–December 8, 2028,
including its own 61-session warmup. It does not borrow protected bars. Evaluation
cannot occur before December 9, 2028; all prerequisites and registration must be
complete by December 8, 2026. Missing that deadline stops this plan rather than
silently moving its dates. This is a specified future option, not a registered
campaign or a recommendation to commit two more years to the project.

The older 2024–2025 data was already used in research. Its improved observation
contract does not restore untouched status or renew a consumed trial. The future
plan cannot produce today's missing evidence. The end-to-end result is **stop at
prerequisites**, retaining the data and plan for review. No evaluator or automation
was installed for a blocked trial.

If later reopened, insufficient support consumes the trial as inconclusive;
adequate support with failed acceptance retires this strategy direction. Passing
research would only justify a separate paper execution test, never automatic live
promotion. Both existing consumed campaigns and report hashes remain unchanged.

## Artifacts and reproduction

All generated evidence is under `artifacts/endpoint_proposal/`:

- `acquisition_absolute.json` and `adjusted_data/*/dataset.provenance.json`:
  finalized governed dataset. Initial relative-path acquisition retained;
  finalization reused cached windows to produce absolute paths for the local audit.
- `adjusted_audit_proposal.json`: original proposal with only audit source adjustment
  changed from split to all; original draft and historical audit retained.
- `endpoint_feasibility.json`: exact-observation validity and remaining blockers.
- `corporate_actions_expanded.json`, `adjustment_samples.json`,
  `adjustment_diagnostics.json`, `sample_dataset_linkage.json`: raw evidence,
  transparent diagnostics and linkage.
- `evaluation_plan_freeze.json`, `decision.json`: frozen plan hash and stop decision.

Reproduce the local audits without network or returns:

```bash
./venv/bin/python -m ai_trading.tools.endpoint_feasibility \
  --proposal artifacts/endpoint_proposal/adjusted_audit_proposal.json \
  --acquisition artifacts/endpoint_proposal/acquisition_absolute.json \
  --output artifacts/endpoint_proposal/endpoint_feasibility.json
./venv/bin/python -m ai_trading.tools.adjustment_diagnostics \
  --samples artifacts/endpoint_proposal/adjustment_samples.json \
  --output artifacts/endpoint_proposal/adjustment_diagnostics.json
```

## Validation and operational limits

`bash scripts/agent_validate_changed.sh --skip-runtime-smoke` passed lint, mypy
across 18 changed source files, compilation and 131 selected tests. This includes
eight new regressions for split/dividend agreement, factor disagreement, invalid
prices, timestamp mismatch and holdout rejection. Log:
`/tmp/endpoint-endtoend-validation.log`. No trading decision or service code changed;
no strategy replay was run because prerequisites stopped evaluation.

A separate live health smoke returned HTTP 503 with existing
`required_model_stale` and `replay_live_parity_gate_failed` flags. No gates were
weakened or service restarted. Rollback removes only the new diagnostic tool/test
and plan/report; retain evidence artifacts and consumed ledgers.
