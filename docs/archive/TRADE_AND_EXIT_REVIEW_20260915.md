# Trade, authority and exit review — September 15, 2026

## Findings and changes

The overnight carry was not the configured policy. Runtime has
AI_TRADING_EOD_FLATTEN_ENABLED=1 and a 300-second lead. At 19:56:42 UTC on September
14, EOD_FLATTEN_TRIGGERED found two positions. AAPL's market-exit submit began at
19:57:23; Alpaca returned HTTP 504 at 20:00:12. APIError escaped exit_all_positions
and the cycle finalizer, crashing the service before MSFT was attempted. Systemd
restarted it after close. This evidence is stronger than the earlier suggestion
that overnight carry might have been intentional.

execution_flow.py now catches ambiguous broker/transport failures per exit,
records EOD_EXIT_SUBMISSION_UNCONFIRMED and continues other positions. Exit client
IDs are stable per New York date/symbol/side across cycles and restarts, preserving
broker idempotency after ambiguous timeouts. Position-list APIError is handled.
No market order was sent by this review. Existing halt, reduce-only and session
rules remain intact. This prevents the observed exception from terminating the
whole closeout; it does not guarantee broker acceptance or completion by close
when an upstream request stalls for minutes. Rejected/canceled same-session exits
require reconciliation; stable identity deliberately avoids new blind replacements.

## Today's observed trading through the 16:57 UTC broker capture

Ten filled orders, each one share. Two sells closed yesterday's positions; four
intraday round trips produced these gross results:

| Symbol | Buy | Sell | Holding minutes | Gross P&L | Exit journal reasons |
| --- | ---: | ---: | ---: | ---: | --- |
| AMZN | 253.14 | 250.07 | 23.56 | -3.07 | FLAT_BEFORE_REVERSAL, OK_TRADE |
| AAPL | 329.82 | 329.55 | 29.30 | -0.27 | OK_TRADE |
| MSFT | 501.92 | 500.00 | 65.28 | -1.92 | FLAT_BEFORE_REVERSAL, OK_TRADE |
| AMZN | 249.13 | 249.08 | 30.07 | -0.05 | OK_TRADE |

Total intraday round-trip P&L: -5.31 before fees. It excludes the carried-position
sells and does not equal the daily account-equity change. Entry quoted spreads
were approximately 3.16, 1.21, 6.18 and 3.21 bps respectively. Fill occurred 10–76
seconds after the recorded decision across all ten trades. The signal bar labels
were 306–395 seconds old at decision; these are five-minute bar start timestamps,
so label age alone does not prove stale finalized features. No parameter tuning,
alternative-strategy scoring or holdout evaluation was performed.

One carried MSFT exit recorded a 113.18-bps decision spread. This deserves quote
lineage review; an indicative stored spread cannot by itself establish executable
cost or justify blocking a risk-reducing exit. Per-fill fees and complete quote
causality remain unverified. The four losses are insufficient to distinguish weak
predictions from exit-policy problems or establish that a different exit wins.

## Explicit strategy and authority attribution

All ten executions join to accepted day-sleeve journals whose debug metadata says
evidence_partition=stale_model_paper_diagnostic, model_authority=False and
runtime_authority=False. Paper sampling allows the diagnostic execution path;
these trades are not evidence that the stale day-sleeve ML model passed its gate.
The stale-model/parity gates and research reset remain unchanged.

The per-trade review at /tmp/four-improvements-trade-review.json explicitly records
strategy, evidence partition, model authority, broker-filled quantity and price,
decision/bar ages, indicative spread and recorded reasons for every fill. The
reproducible read-only script is artifacts/audits/trade_review_20260915.py.

Additional reporting gaps: durable intent strategy_id is null for these trades;
model_id=trained_model.pkl alone is insufficient attribution. Two AMZN decision
receipts retain requested quantities 8/12, while the broker filled one share each
after sampling. The review preserves requested_receipt_qty separately from broker
qty rather than treating those receipts as broker truth. Historical journals were
not rewritten. These upstream reporting gaps are documented follow-ups; the review
does not silently certify them as repaired.

## Deterministic tests

Two daily-fallback tests replaced sys.modules stubs with patches on the actual
imported alpaca_api.get_bars_df. The forced-Yahoo/memo test now mocks its primary
call too and uses a request window consistent with its historical memo fixture.
The implementation currently fetches primary data before consulting the memo;
the test fix isolates that behavior rather than changing runtime caching policy.

Focused suites: 73 passed, including the three formerly failing network tests,
APIError closeout continuation and stable exit IDs. Updated two existing exit
expectation tests for client IDs. No new compatibility shim or SDK change.
Validation/deployment status is in CODEX_HANDOFF.md. Logs are private under
/tmp/four-improvements-*. Rollback is limited to execution_flow.py and matching
tests; the analytical report and historical source records need no rollback.
