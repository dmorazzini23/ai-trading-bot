# Minute gaps and feature skew — September 17 review

Read-only investigation of September 16 logs from 15:04 UTC onward. No runtime
code, data filling, feed, model, gate or threshold changes were justified.

## Minute gaps

187 warnings comprise 54 AMZN and 133 MSFT reports. Their timestamp samples
contain four unique missing minutes, repeatedly detected in overlapping windows:

- AMZN: September 15 at 16:42 UTC.
- MSFT: September 16 at 16:29, 17:13 and 18:17 UTC.

Independent direct Alpaca historical requests using the logged IEX feed and all
adjustments reproduce all four omissions. MSFT's September 16 regular session
returns 387 of 390 expected minute labels. A focused prior-session AMZN request
returns surrounding minutes but omits 16:42. Therefore these are not solely local
cache omissions. This does not establish why IEX omitted the bars or prove every
local fetch/cache path correct. No alternate feed or synthetic bar was substituted.

The direct September 16 AMZN session additionally lacks six later minute labels;
those are not the timestamps responsible for the 54 reviewed AMZN warnings.
Do not infer log coverage for periods when that symbol was not requested.

Evidence: /tmp/gaps-skew-provider.json, /tmp/gaps-skew-amzn-prior.json,
/tmp/change-check-logs.jsonl. Repeated warning count is not a count of distinct
outages, and an upstream absence is not automatically a trading-system defect.

## QQQ feature warning

At September 16 16:50:46 UTC, ML_TRAINING_SERVING_SKEW reported RSI, ATR, SMA200,
ATR percentage and centered RSI outside training percentile bounds. Five of twelve
observed features gives 0.4166667, exceeding the 0.35 outlier-ratio threshold.
Mean absolute z-score was 1.2063, below 2.5; maximum absolute z-score was 2.69375.
The code uses an OR between mean-z and outlier-ratio breaches, so the warning is
consistent with its documented calculation. The correlated derived RSI/ATR
features are separate columns; their count is not independent statistical evidence.

Training statistics are per-feature means/stds/5th/95th percentiles computed over
the supplied training dataset. This warning is a distribution diagnostic, not
proof of bad arithmetic, a live/training feature mismatch, or qualified-model
performance. Historical warning fields omit the individual observed values,
reference statistics and model artifact identity. Consequently the underlying
historical excursion cannot be reproduced exactly from this warning alone.
Re-fetching today's revised bars would not establish yesterday's exact inputs.

No causal root cause beyond the threshold breach is claimed. Stale-model and
replay gates remain in force. A future forensic improvement would retain exact
feature/reference values and model identity; it is not an urgent trading fix.

## Production reporting verification

All eight September 16 durable intents have strategy_id=day and quantity=1.
Decision journals agree on strategy and submitted quantity. Requests of 2, 6 and
10 shares remain distinct from sampled one-share submissions. Receipts capture
pending_new with filled_qty=0; later durable records are FILLED. These are different
lifecycle timestamps, not contradictory quantities. Final broker fills were
independently confirmed in the September 16 end-of-day status check.

## Validation and limitations

Direct broker-data responses and read-only journal/DB joins support the findings.
No new regression tests or service restart were needed because runtime code was
unchanged. Docs-only validator and diff checks passed. Model/feature causal
diagnosis and closeout with remaining exposure are not falsely marked verified.
