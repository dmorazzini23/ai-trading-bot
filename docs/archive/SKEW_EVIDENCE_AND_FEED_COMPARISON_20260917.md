# Skew evidence fix and bounded feed comparison

The skew warning now retains each observed feature value, its training mean,
standard deviation and 5th/95th-percentile reference values. It also records the
last datetime-index row label, a separately recorded capture time, model class,
and a joblib SHA1 fingerprint of the actual in-memory estimator. The row label
is not asserted to be the causal decision time. Missing datetime labels stay
unavailable rather than being replaced by the capture time.

The model fingerprint identifies serialized estimator state, not an artifact-file
checksum or proof of model qualification. Unsupported serialization is explicitly
reported and does not suppress the skew warning. No configured file path is
silently substituted for the estimator actually supplied to inference.

Feature selection, predictions, thresholds, breach arithmetic and qualification
gates are unchanged. The diagnostic payload is emitted on breaches. Model
fingerprinting adds work only on breaches; its production overhead has not been
benchmarked. A naturally occurring warning is still needed to verify production
payload capture after deployment.

## Same-window feed comparison

Historical one-minute requests used identical symbol, regular-session boundaries
and all adjustment settings. Endpoints were filtered to the half-open session
interval, without imputation or cross-feed filling.

| Symbol/session | IEX bars | SIP bars | Missing IEX labels (UTC) |
| --- | ---: | ---: | --- |
| AMZN, September 15 | 389/390 | 390/390 | 16:42 |
| MSFT, September 16 | 387/390 | 390/390 | 16:29, 17:13, 18:17 |

Artifact: /tmp/skew-fix-feed-comparison.json. This confirms that consolidated
historical data fills these observed coverage gaps. It does not establish
real-time SIP entitlement, data latency, identical feature distributions or
executable fills. It is a bounded data-quality comparison, not a strategy test.

No feed or ticker setting was changed. A production SIP migration requires a
verified real-time entitlement, a declared consistent training/serving contract,
and validation of finalized bars/features. The research reset, existing models,
consumed budgets and holdout remain unchanged. No model was retrained.

## Validation and rollback

Validation: initial selected run passed 343 tests but exposed six stale lifecycle
fixtures that mocked away submitted_qty. Those tests now use the real order
normalizer. Final validator passed 20 selected cases (14 skew/signal and six
lifecycle), lint, mypy (3 files) and compilation. Final focused signal run also
passed 14 tests. Non-sending incident check passed. Logs:
/tmp/skew-fix-validation.log, /tmp/skew-fix-validation-final.log,
/tmp/skew-fix-focused.log. No broad full-suite pass is claimed.

Regression coverage checks exact feature/reference values, datetime-label
propagation through signal_ml, identity of the actual estimator fingerprint,
and explicit unavailable identity when serialization fails. No new backtest is
needed because this is diagnostic-only behavior.

Rollback only the added diagnostic hunks in bot_engine.py and restart the
service. There is no schema migration or historical record rewrite.

Deployment: restarted at 03:31:15 UTC September 17. At 03:32:40, health is healthy,
runtime ready, broker connected/fresh with zero positions/open orders. Startup
logs show no warning/error; existing stale-model/parity attention flags remain.
Artifacts: /tmp/skew-fix-posthealth.json and /tmp/skew-fix-startup.jsonl.
Docs-only validation and diff checks passed. Natural breach capture is pending.
