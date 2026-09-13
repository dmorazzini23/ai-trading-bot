# Post-fix replay evidence review

September 12, 2026. Decision: suitable for diagnostics, not model qualification.

## Run and provenance

No post-fix scheduled artifact existed at the start of this review. Starting the
systemd job required an unavailable sudo password. The same replay-governance CLI
was therefore run as aiuser with /run/ai-trading-bot/ai-trading-runtime.env loaded,
using --force and isolated data/output/summary paths under
/tmp/postfix-replay-check. Force bypasses schedule timing, not qualification gates.
No strategy, threshold, seed, cost, source-window or qualification setting changed.
The production replay artifact and live service were not replaced or restarted.

Artifact: /tmp/postfix-replay-check/output/replay_hash_20260912.json

SHA-256: 44f2f0aee4190b2c00723848e7f0da310739dfb17a8d3631b089d4f02ec6bb5a

CLI summary: /tmp/postfix-replay-check/summary.json

Run status: blocked, REPLAY_POLICY_NON_REGRESSION_FAILED. The artifact was written
before the qualification failure and remains usable for diagnostic inspection.

## 1. Contract verification

All 68 candidate markouts have supported order types (all limit), timezone-aware
decision/reference/fill/markout timestamps, measured intervals and fill assumptions.
Submission status is separately not_submitted for 57 and unknown for 11. Unknown
does not imply submitted or broker-confirmed. Every row passed chronological,
scheduled-time, interval and gross-minus-drag consistency checks. The maximum
markout ceiling of 24 hours is respected. This verifies the new field contract;
it does not establish realistic execution or historical submission intent.

All rows disclose simulator assumptions: seed 42, spread 8 bps, volatility 0.01,
fill probability 0.95, partial-fill probability 0.35, zero fee bps, configured
jitter/volatility ranges, observed simulation price and limit constraint handling.
These remain assumed execution behavior, not independently observed costs.

## 2. Record reconciliation

| Stage | Count |
| --- | ---: |
| Shadow records scanned | 20,000 |
| Shadow records accepted before deduplication | 1,826 |
| Shadow records rejected | 18,174 |
| Unsupported or missing order type | 7,396 |
| Parity marker missing | 5,666 |
| Before lookback | 4,199 |
| Opportunity not eligible | 871 |
| Not explicitly unsubmitted | 42 |
| TCA records scanned | 18,717 |
| TCA records rejected | 18,653 |
| TCA records passing initial filters | 64 |
| Final shadow rows | 1,825 |
| Final TCA rows | 34 |
| Final opportunity rows | 0 |
| Refreshed rows after deduplication/replacement | 1,859 |
| Rows in configured AAPL/AMZN/MSFT universe | 911 |
| Candidate submitted simulated orders | 77 |
| Candidate fill events | 74 |
| Candidate qualified-horizon markouts | 68 |

Shadow rejection reasons sum exactly to 18,174; accepted plus rejected equals
20,000. Final source composition sums to 1,859. The 948-row reduction to 911 is
fully explained by the configured symbol universe. All refreshed rows are labeled
5Min. The combined 1,890 initially accepted records reduce by 31 through the
existing identity deduplication and TCA replacement path; counters do not separate
those two effects. Six fill events do not contribute a valid subsequent markout;
the artifact lacks event-level rejection reasons, so their exact exclusions are
not established. Do not count fills as independent trades or all accepted records
as final retained rows.

No opportunity rows with unknown executable order intent were admitted. The
7,396 rejected records are an explicit evidence gap, not a join failure to bypass.

## 3. Observed timing versus model target

| Interval | Minimum | Median | 90th percentile* | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Decision to simulated fill | 46.245 s | 3,600 s | 78,300 s | 341,100 s |
| Fill to markout | 60.159 s | 600 s | 3,900 s | 81,628.233 s |

*Empirical sorted value at floor(0.9 × (n−1)), n=68.

Only 28 markout intervals equal 300 seconds; 36 exceed it. Fifty-four fill delays
exceed five minutes; the maximum is 94.75 hours. Pending simulated limit orders can
therefore survive across sessions in this sparse observation stream.

The selected ml_edge model's registry metadata declares horizon_unit=bars,
horizon_bars=1 and required_bar_timeframe=5Min. Its recorded training-label timing
already ranges from five minutes to about 89.67 hours, with a five-minute median.
Thus a bar-count target did not guarantee fixed elapsed time even during training.
Replay's median ten-minute post-fill interval and potentially multi-day fill delay
do not validate a fixed five-minute economic horizon. No linked exit evidence in
this report establishes an intended realized holding period. This is a documented
horizon mismatch/ambiguity, not permission to select a more favorable horizon.

## 4. Qualification decision and next gap

Candidate: -10.888324 bps across 68 markouts. Baseline: -6.790418 bps across 769.
The positive-edge, 250-sample minimum and edge-non-regression checks fail. Drawdown
non-regression passes. All thresholds remain unchanged. These results must not be
compared as a controlled improvement over the older mixed-order artifact: accepted
evidence and the rolling source window differ.

The report is diagnostic-only because executable-intent coverage is incomplete,
submission status remains unknown for some rows, simulator assumptions lack
execution validation, and elapsed-time/expiry behavior does not establish the
strategy's target holding period. Keep model abstention and qualification gates.

Next corrective scope: explicitly specify replay order expiry/time-in-force and
persist markout exclusion reasons, then validate target-horizon semantics under
the governed protocol. Do not run a horizon search or relax costs to improve scores.

The source interval August 13–September 11 overlaps the protected holdout. This
run used the existing operational evidence workflow only; no subgroup selection,
model fitting, parameter tuning, order submission or promotion occurred. Do not
claim historical holdout isolation. Validation used arithmetic/contract assertions
on the output; documentation validation and git diff --check passed.
