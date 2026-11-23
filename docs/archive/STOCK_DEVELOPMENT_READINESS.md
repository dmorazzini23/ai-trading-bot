# Stock development readiness — September 13, 2026

The governed development dataset now includes AAPL, AMZN and MSFT. This closes
the symbol-coverage gap, not the model qualification or execution-evidence gaps.

## Source and scope

Canonical historical_training_backfill acquired SIP, split-adjusted 1Min bars
for 2024-01-01 through 2025-12-31 into artifacts/stock_development. No holdout
dates were requested. Acquisition and per-symbol quality checks passed: each
stock contains all 194,700 expected regular-session minutes over 502 sessions.
Extended-session source rows are excluded from the readiness assessment.
Acquisition manifest: artifacts/stock_development/acquisition.json.

The new `ai_trading.tools.stock_development_readiness` CLI verifies source identity,
hashes, quality, and read stability using the canonical development-data loader.
The loader was extracted from the existing sampling audit; slower-campaign
sampling behavior remains unchanged. Actual timestamp bounds are also checked.
Output: artifacts/stock_development/readiness.json, including source hashes.

## Findings

| Per stock (same counts for all three) | Count |
| --- | ---: |
| Candidate decision slots | 38,438 |
| Session-boundary exclusions | 502 |
| Complete timestamp / valid OHLCV windows | 37,936 |
| Insufficient feature history | 195 |
| Necessary feature-input support | 37,741 |
| Missing, duplicate or invalid regular-session minute exclusions | 0 |

The proposed timing convention remains D+1 entry, D+6 exit, complete minutes D-5
through D+6, all within a regular session. OHLCV must be finite and positive with
valid high/low geometry. Zero-volume bars are conservatively excluded rather than
assumed executable. Session calendars include early closes.

Feature-input support additionally requires 200 complete five-minute blocks on
the regular-session grid and complete current-session history through decision
time for VWAP. Overnight closures are expected calendar gaps, not missing bars.
No missing market observations are filled. Warmup exclusions are counted before
claiming support; slots are overlapping opportunities, not independent trades.

## What remains unverified

The selected model's 12 feature columns include SMA200, SMA50, RSI, MACD, ATR,
VWAP and their derived values. Its `_feature_frame` performs final ffill/fillna(0)
and `_safe_rsi` can return zeros on failure. Thus finite feature arrays are not
proof of source validity. This audit checks necessary histories and raw geometry;
it does not run or certify full feature computation, numerical convergence,
prefix causality, training/serving parity, or adjustment consistency. EMA history
and the selected model's irregular label horizons still require explicit review.
No imputation policy was changed, no model fitted, and no predictions or returns
were computed. Support here cannot authorize promotion.

Fresh paper broker accounting found 535 activities and 407 matched order
quantities. All 407 compared local fee totals remain unknown; fee activities have
zero execution references. Quantity agreement does not prove fill identity,
complete feeds, actionable quotes, market depth, or execution costs. Historical
OHLCV cannot establish any of those. Reports: /tmp/stock-audit-accounting.json and
/tmp/stock-audit-broker-snapshot.json (private raw account data).

## Validation and operations

The server rebooted at 00:57:01 UTC. Service active; broker connected and fresh,
zero positions/orders, existing stale-model/replay attention flags only. No
trading-service changes or restart were needed for this audit.

Regression coverage checks invalid geometry, non-finite prices, zero/negative
volume, duplicate timestamps, insufficient history and successful 200-bar warmup.
See CODEX_HANDOFF.md for final validator results. No campaign trial was claimed;
holdout and research-reset restrictions remain intact. Rollback removes this
audit and its tests and restores the source-loader extraction if needed; preserve
acquired manifests/data and campaign evidence for auditability.
