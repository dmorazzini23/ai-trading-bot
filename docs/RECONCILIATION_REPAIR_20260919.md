# September 18 reconciliation investigation

## Evidence and root causes

At 15:11:53Z and 15:17:05Z the service logged two quantity mismatches,
maximum absolute delta 3 and ratio 2/3. Replaying the currently retained,
promotion-filtered AAPL/AMZN/MSFT trade history up to either timestamp produces
AAPL -3, AMZN 2, MSFT 1. The session's broker fills from a flat opening instead
imply AAPL 0, AMZN 1, MSFT 1. This reproduces both logged discrepancy magnitudes.
This is a retrospective replay of retained records, not an immutable snapshot
of the exact inputs read by the process at each incident.

The 15:26:50Z report retained reconstructed AAPL -3, AMZN 1, MSFT 1 but claimed
zero mismatches using `broker_open_positions` as both comparison sides. The
existing large-discrepancy fallback concealed differences instead of resolving
them. Removed that fallback; legacy fallback environment settings no longer
authorize substituting the broker's answer for independent ledger evidence.
The earlier ratio2/3 was below the default0.8 fallback threshold. At15:26 the
independent ledger still has two mismatches with total absolute delta4 and
ratio0.8; this crossed the fallback threshold and explains the apparent recovery.
Replaying that saved report through the corrected comparison reports both
mismatches rather than zero.

Two additional bugs were confirmed with regressions:

- FIFO reconstruction overwrote a symbol's residual quantity when multiple
  lineage books remained open. Sum the books, including offsetting quantities.
  This change does not alter the reproduced September 18 discrepancies and is
  not claimed as their historical cause.
- The audit-to-meta converter emitted duplicate opposite-reward records for a
  single round trip and mishandled partial quantities. Consume FIFO entry lots,
  emit one observation per matched quantity and preserve only unmatched residual
  inventory. Validate finite positive quantities/prices and supported sides.
  No evidence establishes this converter as the source of the runtime blocks;
  the performance reporter has its own reconstruction path.

Historical records were not rewritten, filled in, or certified complete. The
ledger discrepancies remain unresolved evidence and must remain blocked.

## CI and release state

Main CI35301288534:6819 passed,8 failed,4 skipped; coverage78.95%.
The eight failures shared sklearn estimators replaced at collection time by
tests/test_integration_robust.py. Preserve real sklearn modules rather than
installing fake estimators that other collected modules permanently retain.
The 56-test sequential reproduction group passes after this change.
Full-dependency run35351358531:6827 passed,4 skipped; coverage78.98%, below80%.
Its retained coverage XML reports121559 covered of153905 statements, requiring
1565 additional covered statements at the same denominator to reach80%.
Coverage gate and qualification thresholds remain unchanged.

Integrated committed close-recap/health changes from bde5a6a4f with remote main
in the isolated worktree. Production checkout and installed dependencies were
not changed during this investigation. Live service active,NRestarts0, started
September18 at06:47Z; broker fresh,zero positions/orders; required_model_stale
still blocks readiness. Do not infer loaded code solely from checkout HEAD.

## Validation and rollback

Reporting/conversion regressions:104 passed. CI-isolation reproduction:56 passed.
The same isolation group also passes with two pytest workers. Final converter
and related regressions:45 passed; integrated replay/health/recap:107 passed.
Counts overlap. Non-sending incident snapshot and live health smoke completed.
Detailed validator/replay/incident outcomes are recorded in CODEX_HANDOFF.md.
Final `bash scripts/agent_validate_changed.sh --skip-runtime-smoke`: lint,
types(eight files),compile and108 regression tests passed. An initial run emitted
imported dataclass-slots diagnostics; final normal run passed, as did clean
baseline source checking and scoped changed-file checking. No suppressions added.
Logs:/tmp/sep19-{all-boundaries,ci-regression,validation,integrated-regression,
incident-smoke}.log. No order, notification, training run or new research trial.

Runtime risk: stricter reconciliation can expose previously concealed stale
inventory and block more entries. That is intentional; restore trusted ledger
evidence rather than suppressing the guard. Converter changes affect future
conversions only. Roll back the relevant source/test patch together if needed;
no data migration or historical artifact mutation needs reversal.
