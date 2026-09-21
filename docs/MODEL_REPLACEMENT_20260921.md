# Bounded model replacement development

The user authorized one bounded replacement-model trial on September 21, 2026.
`config/model_replacement_campaign.json` freezes its specification. Registration
is persisted in `artifacts/model_replacement_20260921/campaign_state.json` with
contract hash `7e41c0bb3b658ebc9e1f21fbf453098fc7aaf3490a66c5a9268f3b19b63f68e5`.
The single trial is complete: `hypothesis_rejected`. Existing consumed ETF campaign
budgets, scheduled-training pause, and promotion gates are unchanged.

## Completed outcome

Built 113,223 observations (37,741 per symbol) from the verified 2024–2025 files.
Five expanding, purged folds evaluated 94,200 common opportunities, selecting
30,802 historical proxy trades. Every fold exceeded 250 selections, so this is a
supported rejection rather than insufficient sample support.

| Metric | Result |
| --- | ---: |
| Net bps per common opportunity, including abstentions | -3.197464 |
| Whole-session bootstrap 95% lower bound | -3.445699 |
| Profitable folds | 0 / 5 |
| Gross bps per selected proxy trade | +0.221377 |
| Net bps per selected proxy trade at fixed 10 bps cost | -9.778623 |
| Always-long net bps per identical opportunity | -9.947166 |

The classifier loses less than always-long by abstaining, but still loses to cash.
This is development evidence, not live P&L, untouched performance or executable
fill evidence. No serving artifact was saved or registry entry created. The
trial budget is consumed; do not rerun or tune this campaign to rescue the result.
The stale-model readiness failure remains because this trial did not establish
a qualified replacement.

Artifacts in the production checkout: `artifacts/model_replacement_20260921/`.
`report.json`, `oof_predictions.parquet` and `campaign_state.json` have matching
verified hashes. Independent checks reproduced cost subtraction, selection,
opportunity means, unique decisions and five-fold coverage. The final service
check found an active service, fresh broker and only required_model_stale; no
restart or model activation occurred.

The audit-time cache bug was corrected before fitting. Before reusing partitions,
the audit compared feature, validation, split and evaluation function ASTs against
57a96e24f, checked unchanged dependency and source hashes, and revalidated every
row. Original manifests and preflight remain as `*.derivation_manifest.json` and
`derivation_preflight.json`; `cache_recertification.json` records the correction.
The low-priority MSFT worker used identical builder functions from 4861beec3;
each current manifest records its partition execution revision. No data or model
outcome was changed by recertification. After the cache correction, eleven focused
tests, lint and mypy passed; the previously completed 663 selected tests remain
the broader validation result.

## Verified preflight

The canonical stock feature audit revalidated acquisition metadata, source hashes,
2024–2025 date bounds, complete regular sessions and valid OHLCV before computing
features. No September–December 2026 holdout prices were read.

All nine fixed AAPL/AMZN/MSFT comparisons passed same-history equality and prefix
causality. All nine also passed the new comparison against a ten-calendar-day
runtime window at the first finalized decision (five-minute close plus two-second
grace), with relative and absolute tolerance 1e-10. At the 1,000-bar prefix this
window contains 545 bars. This checks the default window, not every possible
deployment override, feed, decision delay or historical observation.

The old 200-bar truncation check fails all nine comparisons. It demonstrates
indicator initialization sensitivity, but is not evidence of a defect in the
default ten-day runtime window. No indicator or inference behavior was changed.
The added regression explicitly distinguishes these two cases.

## Evaluation implementation

Runner validation on the integrated main repair history: ten focused tests and
663 standard-validator selected tests passed; changed-file lint, mypy and
compilation passed. The validator's sandboxed localhost check could not connect;
a direct health check confirmed active service, fresh broker, no open positions
or orders and the existing required_model_stale failure. The non-sending incident
snapshot check passed. Logs: /tmp/replacement-trial-tests-final.log and
/tmp/replacement-standard-validation.log. These results supersede the older
production-checkout validation failures recorded at the bottom of this report.

`ai_trading.tools.model_replacement_trial` builds one hashed Parquet partition per
symbol, using canonical feature construction separately at each finalized decision.
It verifies complete regular sessions (including early closes), source hashes and
development bounds. Reuse requires matching source, contract, code and output
hashes. Dataset construction does not fit or select a model.

Decision time is five-minute bar start plus five minutes and two seconds. Entry
is the next minute open and exit is five minutes after entry, strictly before the
session close. All supported five-minute opportunities are retained; adjacent
entries can coincide with the previous exit but cannot overlap. Labels subtract
the frozen ten-basis-point cost. Bars are proxies, not executable fills.

The implementation splits the ordered development sessions into six contiguous
chunks with numpy.array_split. The first chunk supplies initial training; each of
the next five is one test fold, with expanding prior training. It removes the
entire immediately preceding session and purges training labels at that boundary.
Scaling fits inside each training fold only. Each fold fits the fixed classifier
once, using probability >= 0.5. No calibration, search or threshold selection is
performed. Whole-session bootstrap resamples all symbols together. Cash is zero;
abstentions remain zero-valued slots in the common-opportunity denominator.

The original campaign registration must exist. `--fit` claims it atomically
immediately before fitting. Any existing claim blocks further data processing or
fitting. Failure after claim requires audit, not automatic retry. The runner does
not save a serving model or update a registry. Even a passing screen remains
`not_qualified` until existing independent qualification gates are evaluated.

```bash
./venv/bin/python -m ai_trading.tools.model_replacement_trial \
  --repository-root /home/aiuser/ai-trading-bot
# Once preflight and regression checks pass, use the same command with --fit.
```

## Original preflight boundary

The registered evaluation needs a derived dataset with hashed provenance for
runtime-equivalent rolling feature windows, finalized decision timestamps,
strictly subsequent entry opens, same-session exits and label end timestamps.
The one-minute acquisition and nine sampled parity checks alone do not establish
that dataset. The existing broad trainer must not be substituted: its selection
and label assumptions do not implement this frozen trial contract.

That dataset and the five-fold purged runner are now implemented and validated.
Only then claim the single trial immediately before fitting. A crash after claim
requires audit; a failed/inconclusive result does not authorize parameter search.
Historical opens with assumed costs remain proxies, not executable fill proof.
Even a passing development result cannot activate a model without the independent
qualification and promotion gates. The stale-model readiness failure remains.

## Validation and runtime scope

Three focused feature-parity tests passed, including the added rolling-window
regression. Changed-file lint, mypy and compilation passed. The standard market
hours validator also selected the broader tests/tools directory: 635 passed and
four failed (two repository-audit tests, momentum feature causality, and training
accelerator candidate handoff). These failures were not repaired in this preflight
and validation is not fully green. Log: /tmp/sep21-model-preflight-validation.log.
No service restart, model activation, live order,
holdout evaluation or training occurred.
