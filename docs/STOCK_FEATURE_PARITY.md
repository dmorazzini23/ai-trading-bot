# Stock feature parity — September 13, 2026

The current replay-training and canonical day-sleeve runtime builders agree on
identical clean development histories in nine fixed checks. This is bounded
formula evidence, not full deployed-model parity or execution validation.

## Reproducible diagnostic

Run `./venv/bin/python -m ai_trading.tools.stock_feature_parity`.
The CLI verifies the governed AAPL/AMZN/MSFT acquisition for 2024-2025, validates
complete regular-session minute OHLCV, then aggregates exact five-minute bars
with open/maximum high/minimum low/close/summed volume. It refuses to compress
missing sessions. Source hashes are checked before and after analysis.

Fixed prefixes: first 250, 500 and 1,000 completed regular-session bars for each
stock. Each check compares all 12 runtime feature columns with `_feature_frame`
at the same prefix, and compares the earlier training row after appending ten
future bars. A separate diagnostic computes the runtime row on only the last
200 bars. Numerical equality uses rtol=atol=1e-10; no model is loaded or scored.
Results: artifacts/stock_development/feature_parity.json, including source hashes.

## Findings

- All nine identical-history comparisons pass.
- All nine prefix-causality checks pass. This is not an exhaustive causality proof.
- All nine truncated-history comparisons differ in RSI, MACD, signal,
  macd_signal_gap and rsi_centered at the stated tolerance. No prediction or
  economic impact was measured; numerical mismatch alone does not establish it.
  The 200-bar truncation is a diagnostic, not the current live fetch policy:
  `_sleeve_bar_fetch_start` uses a configurable calendar-day lookback (default 10,
  bounded to 7-60). This audit does not certify the effective provider response.
- A synthetic 20-bar regression demonstrates the fallback boundary: training
  produces finite values after imputation while runtime rejects the non-finite
  feature row. Imputed output must not count as valid warmup evidence.

## Remaining contract requirements

200 complete bars support SMA200 but do not define identical initialization for
recursive indicators. Full parity additionally requires the same history start
or verified indicator state, identical session and adjustment conventions, bar
finalization, aggregation and feature-column order. The checks here deliberately
use identical sources; they do not verify live-provider histories against the
historical training inputs or reconstruct the selected model's original build.

No training fallback, live builder, model artifact, threshold or qualification gate
was changed. Do not retroactively adopt a new history convention for the selected
model. Its stale status and irregular label-horizon issue remain. Broker fee and
execution-evidence gaps reported in STOCK_DEVELOPMENT_READINESS.md also remain.

## Validation

New regressions exercise identical-history parity, prefix causality and the
training/runtime warmup rejection difference. The standalone audit completed on
all three stocks. Campaign-ledger hashes remain unchanged; no trials, returns,
predictions, fitting, holdout evaluation or orders occurred. Health showed a fresh
broker with existing stale-model/replay flags; non-sending incident check passed.
See CODEX_HANDOFF.md for final changed-file validation results. No service restart
is required for this diagnostic tool. Rollback removes only the tool, tests and
documentation, preserving saved source evidence and campaign ledgers.
