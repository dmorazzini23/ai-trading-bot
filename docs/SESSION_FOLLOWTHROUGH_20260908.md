# September 8 session follow-through

The requested four actions were pursued during the open paper session. Closing
evidence and unavailable broker documents cannot be manufactured or inferred.

## Session capture

The opening window contains 15 paper broker snapshots, from 13:15:33 through
13:29:50 UTC, and capture continued during the session. The existing user timer
is active for 23:00 UTC September 8. Full-session reconciliation must wait until
after 20:15 UTC, including the closing window. The current non-sending review is
`artifacts/session_followthrough/latest.json`, explicitly awaiting session close.
No trades were forced to create samples.

## Timestamp correction

Both canonical decision recorders previously defaulted `decision_ts` to the
source bar timestamp. New records retain `source_timestamp` and `bar_ts`, add
`recorded_at`, and default `decision_ts` to recording time. `decision_ts_basis`
distinguishes `explicit` from `record_capture`; supplied decision timestamps
are preserved. Historical rows are not rewritten.

The research quote audit separately counts alignment to an explicit decision,
alignment only to record capture, and unverified alignment. A recording-time
fallback does not certify the time a trading decision was made, execution fees,
or causal fill comparisons. The correction prevents bar time from silently
masquerading as observed decision time.

The paper service was restarted after confirming connected/fresh broker state,
zero positions and zero open orders. Trading eligibility and model gates were
not relaxed. Six new runtime records verified the change: recording timestamps
around 14:38 UTC remain separate from the 14:30 source bars. The audit also uses
the elapsed time from quote timestamp to record capture for freshness, preventing
an earlier cached age measurement from understating the quote's actual age.

## Fee evidence

The 14:36 UTC account refresh returned 535 activities: 454 fills and 81 fees.
408 historical order quantities match, but all 408 still lack verified per-fill
total fees. Aggregate fees remain unallocated. The configured paper Trading API
does not provide the separately permissioned Broker document endpoint's transaction
confirmations. This is an unresolved evidence dependency, not proof of zero fees.
Current sources and reconciliation are saved under `artifacts/session_followthrough/`.

## Serving-model review

The selected model was trained July 17 and is roughly 53 days old against a
14-day maximum. It remains shadow/paper-only and abstains. The latest reviewed
registry output contains 12 named candidates, rejected for inadequate development
evidence, samples, cost-adjusted expectancy and replay/shadow support. The latest
training attempt reports no qualified candidate. The separate opening-reversal
study was inconclusive and grants no serving authority.

There is no qualified replacement to install. Refreshing the date, substituting
an unqualified candidate, or weakening freshness would not resolve that problem.
Model availability therefore remains blocked on qualifying strategy evidence.

## Validation and rollback

The market-hours changed-file validation passed lint, mypy, compilation and
372 selected regression tests. A focused journal/recorder/integration run passed
34 tests, and the updated quote-audit assertions passed three tests. Coverage
checks preserved source time, recording-time fallback, explicit decision times,
and distinct audit treatment of capture versus decision alignment.

Rollback reverts the two recorder changes and quote-audit classification, then
restarts the service after an exposure check. Preserve generated evidence. The
known stale-model and replay-parity health failures are separate from this fix.

Final 18:22 UTC verification: service active, broker connected/fresh, zero open
orders and positions, and complete capture snapshots continuing through 18:22:14.
All of the latest 100 decision records contain the new timestamp basis. Health
remains HTTP 503 solely with the existing stale-model and replay-parity attention
flags. The refreshed audit still reports no verified decision-aligned quotes:
timestamp recording is corrected, but causal execution-cost evidence remains
unverified. The final freshness regression suite passed three tests plus lint
and mypy. The full-session result remains pending until after market close.
