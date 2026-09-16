# Deep input validation fixes — September 13, 2026

Four code issues are corrected. The fifth item, execution evidence, was refreshed
and remains blocked on missing source evidence. No existing model, research
budget, cost threshold, feed setting or promotion gate was changed.

## Contract and live input validation

Input contracts require supported scalar values and explicitly structured history
and finality policies. Empty dictionaries, arrays, booleans, non-finite numbers,
unsupported policy kinds and invalid bounds cannot match. Whole calendar-day
lookbacks must be 7-60; grace must be a finite numeric 0-60 seconds. Feature
version and timeframe must match supported declarations. Unknown remains unverified.

The live builder validates the entire OHLCV history before indicators: unique,
ordered, timezone-aware five-minute-grid timestamps, no within-date cadence gaps,
finite positive prices/volume and consistent high/low geometry. This intentionally
rejects zero-volume histories instead of treating them as executable evidence.
Session-calendar filtering remains upstream. Missing dates across sessions still
require separate acquisition-completeness evidence; these checks do not certify it.

## Training features and label timing

Replay training no longer forward-fills or zero-fills feature values. Missing
features remain NaN and are excluded from training eligibility. RSI computation
errors/alignment failures propagate instead of becoming synthetic zeros in both
replay and after-hours training. Feature-cache schema changed to
replay_aligned_features_v2_no_imputation, preventing old imputed cache reuse.

Both training paths share training/label_timing.py. An h-bar label requires h
consecutive five-minute intervals in the same regular exchange session. Missing
intervals, overnight jumps, holidays and post-early-close endpoints are excluded.
Shadow override labels must match the declared elapsed horizon and exchange
session; imported labels cannot bypass timing rules. This formalizes the existing
five-minute bar-count target, not the separate proposed D+1/D+6 execution convention.
No return objective, fee/slippage assumption, horizon search or old label rewrite
was introduced. Minute-frequency inputs cannot masquerade as five-minute labels.

Reports retain all quarantine counts and feature/timing reasons, including when
no eligible rows remain. Expected initial SMA200 warmup and cross-session endpoint
exclusions are reported separately from unexpected within-session defects. The
existing numeric invalid-data threshold is unchanged and applies to the remaining
eligible population; structural exclusions are never counted as usable samples.
After-hours training logs its input-eligibility counts before returning rows.

## Execution evidence review

Read-only paper activity refresh: 535 activities, 407 matching order quantities,
407 unknown compared per-fill fee totals. The report explicitly states
unavailable_without_complete_per_fill_totals. No costs were inferred from absent
records or allocated from account-level fees. Raw snapshot and report are private:
/tmp/deep-input-broker-snapshot.json and /tmp/deep-input-accounting.json.

Full execution proof still requires attributable fills, causal quote/decision
lineage and complete fee evidence. This cannot be manufactured by a code change,
simulated fill, or forced trade. No order was submitted by this work.

## Verification and risk

Focused contract/live/replay-training regressions: 46 passed. After-hours training
and helper regressions: 114 passed. Synthetic artifact/control-flow tests now use
deterministic qualification metrics, while direct model/evaluation tests remain
separate. Synthetic training bars include explicit warmup and real five-minute
spacing rather than relying on imputation or one-minute labels.

The governed development feature audit still passes 9/9 identical-history and
9/9 prefix-causality checks. No real model was fitted or scored; campaign-ledger
hashes are unchanged. Final validator and runtime recovery are in CODEX_HANDOFF.md.

Stricter inputs can cause additional abstention and reduce future training sample
counts; that is intentional. No automatic replacement of the selected stale model
is justified. Rollback reverts these scoped validation/training edits and tests;
preserve model artifacts, acquisition evidence and consumed campaign ledgers.
