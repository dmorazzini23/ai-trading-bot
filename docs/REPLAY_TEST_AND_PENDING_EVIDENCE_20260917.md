# Replay regressions and pending evidence — September 17

## Four replay failures resolved

The failures were stale fixture/assertion assumptions about limit fills, not a
demonstrated runtime execution defect. Ordinary submitted limits are eligible
on subsequent observed prices. End-of-replay draining has no new market price
and cannot manufacture a fill. Buy limits in monotonically rising observations
can remain unfilled even with fill_probability=1.

Changes in tests/test_offline_replay.py:

- Netting reduction: submit once, then supply a later equal price; retain the
  original long and short reduction, no violation and no cap-adjustment assertions.
- Markout metrics: add optional plateaus to the synthetic rising-price fixture
  so limits actually fill on later observations and markouts remain testable.
- Opening-only policy: equal subsequent prices allow the opening fill before
  checking repeat-entry policy. Retain expected skip and acceptance counts.
- Duplicate timestamps/model scoring: assert accepted candidates and no scoring
  error; explicitly expect zero fills for the rising-price fixture.

All 30 tests in tests/test_offline_replay.py passed. No production fill timing,
pricing, risk, expiry, eligibility or qualification behavior was changed.
The changed-file validator also passed: 223 tests, lint, types (seven files),
compilation and forbidden-pattern checks. Live health was checked separately.

## Runtime evidence

At approximately 03:55 UTC the service was active, NRestarts=0, health healthy,
with existing required_model_stale and replay_live_parity_gate_failed flags.
The available post-deployment journal contained 123 lines; no skew, closeout or
error matched the bounded search. This is absence of a new observed event, not
successful natural verification. Market is closed. Natural skew diagnostics
and closeout with exposure remain pending; no trades/warnings were forced and
no background monitor was installed.

## Compatibility remains unverified

Rechecked original July17 artifact manifest and matching original training
report. They declare IEX but neither includes a full versioned input contract
or adjustment/session/history/finality declarations. Exercised the canonical
comparison using the manifest's absent input_contract and a complete illustrative
serving contract: status=unverified, version_matched=false,
qualification_authority=false. The missing contract produces seven missing
fields; this does not erase the separate partial IEX/5Min provenance declarations.
No artifact metadata or gate was changed. More evidence is required to establish
compatibility; lack of paid SIP is not grounds to infer it.

Evidence: /tmp/three-replay-tests.log, /tmp/three-followup-journal.log,
/tmp/three-followup-health.json, /tmp/three-compatibility-result.json.
Rollback: revert only the test hunks from this follow-up. No restart or migration.
