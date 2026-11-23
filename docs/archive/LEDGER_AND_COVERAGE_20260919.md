# Broker ledger reconstruction and test isolation

Removed import-time fake-module installation from test_health.py,
test_bot_extended.py and test_integration_robust.py. These tests now use installed
dependencies and their existing per-test monkeypatches. This removes the shared
sklearn mutation causing the nine failures in CI35409703913; it also removes
similar collection-time mutation of Alpaca, Flask and metrics dependencies.
No runtime code was excluded from coverage and no coverage threshold changed.

The existing broker_accounting_evidence CLI now optionally accepts
--opening-positions, --closing-positions and --ledger-output together. It writes
a separate quantity ledger, requires complete same-account paper boundaries and
broker activity pagination covering the interval, rejects malformed/conflicting
execution identities and preserves original inputs. Acknowledging quantity
agreement does not authorize fee estimates or promotion.

Generated artifacts (ignored by git) are in the production checkout under
artifacts/ledger_rebuild/20260918/:

- ledger.json: eight September18 executions, matched opening-to-closing quantities.
- extended_ledger.json:50 executions from the earliest available complete anchor,
  September7 at03:19:26Z, through September18 at20:01:12Z, matched quantities.
- history_diagnosis.json: the legacy ledger already has AAPL-3/AMZN+1 at the
  September18 flat broker opening. This establishes a pre-session discrepancy,
  not its exact historical origin.
- opening.json,closing.json,extended_opening.json and accounting.json preserve
  the audit inputs/results. Broker activity source remains the immutable daily
  capture under runtime/research_reports/daily/20260918T203902Z_daily/.

History preceding the verified anchor is explicitly outside the rebuilt ledger.
It is not repaired or certified. The rebuilt ledger is not automatically wired
into live reconciliation or training; original runtime history remains unchanged.
Research reset and qualification gates remain in force. No restart/deployment.

Validation:66 isolation/training regressions passed with two workers;31 initial
data-schema/accounting tests passed;17 final ledger/CLI tests passed; standard
changed-file validator passed lint,types6,compile and40 tests. Counts overlap.
Live health and non-sending incident snapshot checked separately. Final test-only
CLI additions passed lint,types and compilation before publishing.

Coverage work adds price/schema preservation, invalid-bar rejection, and ledger
integrity/boundary/CLI tests. Focused measurement exercised38 previously missed
statements in unchanged modules versus the prior full-dependency baseline;
this is not an overall percentage or a promise to clear80%. Use one full CI run
to measure the actual result. Logs:/tmp/efficient-*.log; focused coverage JSON:
/tmp/efficient-target-coverage.json. Remaining coverage requires a larger effort;
avoid repeating full runs or arbitrary tests solely to inflate the metric.

Rollback source/test changes together; generated audit artifacts can be retained
because they have no operational authority. No original ledger was mutated.

## Completed full-run measurement and follow-up

CI35411547532 on4ae076049 measured78.99% (153911 statements,32332 missed):
6880 passed,4 failed,4 skipped. Previous comparable main CI was78.95% with nine
sklearn failures; all of those failures disappeared. Coverage remains below80%.
The four remaining failures were test isolation defects: inherited logging
throttle state, a real network request in the minute-fallback logging test,
reused position-sizing cache and shared order-monitor state. Isolate those
inputs/resources; do not weaken their behavioral assertions. Follow-up22 tests
pass with two workers, and the standard validator passes lint,types4,compile
and nine mapped tests. Runtime code is unchanged by this follow-up.

The publishing-triggered CI for the final test-only patch still needs its own
result. To conserve the user's remaining allowance, no further broad coverage
campaign is included here. Do not report that the final full suite is green or
that coverage reached80%. Full log:/tmp/efficient-full-ci.log; follow-up logs:
/tmp/efficient-followup-{final,validation}.log.
