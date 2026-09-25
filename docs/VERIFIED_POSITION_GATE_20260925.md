# Verified position gate, September 25

The runtime go/no-go position check now requires a recent, bounded paper-broker
quantity audit. The old unbounded `trade_history.parquet` reconstruction remains
in `diagnostic_open_position_reconciliation`; it cannot satisfy the gate. Its
AAPL -3 / AMZN +1 legacy mismatch remains visible and unresolved. No source
history or opening balance was changed.

## Evidence contract

The optional `--verified-position-evidence-path` JSON contains the raw
`snapshot`, `opening`, and `closing` objects. The report rebuilds the ledger;
it does not trust a precomputed match flag. It requires complete pagination,
timezone-aware and ordered times, complete same-account paper position
boundaries, distinct broker fill IDs with valid order/side/quantity/times, no
unaccounted position-changing activity, and exact opening + fills = closing
quantity. The closing boundary must be at most 60 seconds old. A separate
current broker position read must identify the same account, occur after the
activity capture, and agree with the closing positions. Any missing, stale,
conflicting, mismatched, or unreadable evidence leaves
`open_position_reconciliation.available=false` and the required gate blocked.
The report records the bundle path and SHA-256 for review.

For an after-close check, copy one recent, complete paper account record from
`runtime/broker_position_boundaries.jsonl` into a separate opening JSON file.
Run `ai_trading.tools.broker_accounting_evidence` with `--fetch-paper`,
`--capture-closing-positions`, `--opening-positions`, `--closing-positions`,
`--snapshot`, `--fills`, `--output`, `--ledger-output`, and
`--position-evidence-output` paths. It captures a fresh closing position and
then broker activities beginning before the opening. Pass the bundle path to
`ai_trading.tools.runtime_performance_report --verified-position-evidence-path`
immediately, with `--go-no-go` if a decision is required. These are read-only
broker calls; keep generated evidence in a restricted runtime or temporary
directory. An old bundle will fail the freshness check.

At 02:48 UTC, a read-only current capture rebuilt a matched flat-to-flat
paper interval with zero broker executions. The report applied its bounded
position check with zero mismatches, while the separate legacy diagnostic
still showed AAPL -3 / AMZN +1. This verifies the code path at that moment;
it does not establish execution-feed completeness in all future intervals,
per-fill fees, net profitability, or model/replay qualification. A later
uncovered activity gap or account change requires a new capture. The
September 9–December 8 holdout and consumed experiment budgets are untouched.

## Acceptance and remaining work

The position gate is resolved only for a fresh, complete interval whose
current broker check agrees. Until another such capture is supplied, routine
runtime reports fail closed on this check. The pre-September 7 history remains
excluded from verified quantity reporting until an earlier same-account
opening and complete execution interval can be established. Other go/no-go,
model, replay, cost, freshness, provenance, and promotion gates are unchanged.
