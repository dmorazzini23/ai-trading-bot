# Session log diagnostics — September 14, 2026

Corrected the misleading FETCH_MINUTE_STALE_USING_ORIGINAL warning to
FETCH_MINUTE_STALE_REJECTED. The existing DataFetchError still rejects the data;
no stale-data acceptance or gate change was introduced.

Minute-gap diagnostics now include provider/feed, window bounds and up to ten
missing timestamps. Skew diagnostics now identify outlying feature names without
changing calculations, thresholds or model authority. These are diagnostic
improvements, not a claim that data gaps or distribution shift have disappeared.

Validation: 49 focused tests passed. Changed-file lint, mypy (22 files), and
compile passed. The larger selected suite had 648 passes and three unmocked Alpaca
DNS failures in daily-fetch tests; get_daily_df is AST-identical to HEAD.
Failures: test_get_daily_df_uses_backup_when_columns_missing,
test_get_daily_df_normalizes_yahoo_regular_market_schema, and
test_get_daily_df_forced_yahoo_and_fresh_memo. Non-sending incident snapshot passed.
Logs: /tmp/log-fix-tests.log and /tmp/log-fix-validation.log.

Service restarted at 18:36:30 UTC as requested. Fresh broker connectivity and
ready service state returned by 18:37 UTC, with two existing positions and no
open orders. Expected stale-model and replay-qualification flags persisted.

After restart the gap logs identified AMZN IEX bars missing at 16:35, 17:56 and
17:59 UTC. A direct same-feed StockHistoricalDataClient query over
16:34–18:01 returned 85 bars, omitting all three timestamps. These gaps exist
in the provider response; their underlying exchange/data-provider cause remains
unverified. No synthetic bars, subscription changes or threshold relaxation.

The earlier stale quotes and AMZN skew warning remain conditions to monitor.
Absence in a short post-restart window cannot prove permanent resolution. The
stale-data rejection branch is verified by regression test; deliberately stale
data was not injected into the live process. No new model or research trial ran.

Full post-restart observation results are recorded in CODEX_HANDOFF.md.
Rollback the scoped log payload/name edits and matching tests if necessary;
runtime decision behavior remains unchanged. Historical data was not modified.
