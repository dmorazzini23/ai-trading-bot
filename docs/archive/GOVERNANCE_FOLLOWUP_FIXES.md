# Governance evidence follow-up — September 11, 2026

Repeated shadow starts previously overwrote metrics and could race with session
updates. Initialization now acquires the same per-model lock as updates. An
existing readable artifact is preserved byte-for-byte; repeated starts are no-ops
and do not rewrite registry status. Unreadable artifacts fail without replacement.
No destructive reset API was added.

Promotion eligibility now requires a nonempty observation identity and valid
SHA-256 digest for every counted shadow session. Legacy, incomplete or malformed
identity coverage fails `shadow_observation_provenance`, even when all financial
and freshness checks pass. These hashes establish update identity, not independent
proof of the source data. The existing explicit `force=True` administrative bypass
still bypasses eligibility; this patch addresses ordinary governed promotion.

Regression coverage includes repeated starts after evidence collection, waiting
on an in-progress writer, unreadable evidence preservation, complete and incomplete
identity coverage, and denial of ordinary promotion with otherwise passing legacy
aggregates. Positive financial-gate fixtures now include synthetic test identities.

## Existing runtime blockers

The September 11 02:52 UTC health snapshot reported fresh broker connectivity,
zero positions and zero open orders. Replay schema, policy hash, freshness and OMS
checks passed. The counterfactual positive-net-edge check failed: candidate
-18.6469865371 bps across 3,192 replay samples, against a 0 bps floor and 250-sample
minimum. This is a negative replay result, not missing replay evidence or measured
live trading performance. Source: /tmp/governance-followup-health-before.json,
referencing /var/lib/ai-trading-bot/runtime/replay_outputs/replay_hash_20260910.json.
No experiment or source-data replay was run during this follow-up.

The selected model is about 55 days old against a 14-day maximum. It remains
abstention-only. The research reset, consumed budgets and holdout remain in force;
neither retraining nor weakening thresholds is authorized by these bug fixes.

## Risk and rollback

Legacy evidence must be reviewed before ordinary promotion. Repeated start calls
no longer reset metrics or change a model back to shadow. Operators needing a new
study should use a distinct registered candidate and authorized study identity.
If this patch regresses behavior, revert only the follow-up changes in
ai_trading/governance/promotion.py and its tests; preserve earlier audit fixes and
unrelated working-tree changes. Final validation and deployment are recorded in
docs/CODEX_HANDOFF.md.
