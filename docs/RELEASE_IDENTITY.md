# Release identity preflight

**Status (2026-09-23):** the read-only verifier and packaged unit checks are
implemented locally. This unit has not been installed, and no CI-approved
deployment candidate has been verified. It grants no live-trading authority.

The preflight compares one reviewed release specification with the current
checkout, effective sanitized configuration, launch profile, applied OMS
migration, migration head and exact configured model file. It returns nonzero
and writes a blocked report if any field is missing or differs. The database
query is read-only, and the report contains hashes and revision IDs but no
database URL, credentials or model bytes.
The specification accepts only the documented fields, so accidental extra
values are rejected rather than copied into the report or recovery bundle.

Example release specification (replace every placeholder from reviewed CI,
configuration and model-promotion evidence):

```json
{
  "schema_version": 1,
  "tested_commit_sha": "<40-character CI-passed commit SHA>",
  "resolved_config_hash": "<tested sanitized config hash>",
  "launch_profile_hash": "<tested launch-profile hash>",
  "schema_revision": "20260506_0001",
  "model_artifact": {
    "path": "trained_model.pkl",
    "sha256": "<approved artifact SHA-256>"
  }
}
```

Run as `aiuser` from the checkout with the exact environment that the packaged
service will use. First verify code/config/model and a readable existing schema
before migration:

```bash
./venv/bin/python -m ai_trading.tools.release_identity \
  --pre-migration \
  --spec /var/lib/ai-trading-bot/runtime/release_spec.json \
  --models-root /var/lib/ai-trading-bot/models \
  --output /var/lib/ai-trading-bot/runtime/release_identity_pre_migration.json
```

After the reviewed migration, require the applied revision to match the
tested checkout's migration head:

```bash
./venv/bin/python -m ai_trading.tools.release_identity \
  --spec /var/lib/ai-trading-bot/runtime/release_spec.json \
  --models-root /var/lib/ai-trading-bot/models \
  --output /var/lib/ai-trading-bot/runtime/release_identity_preflight.json
```

The expected commit must have the required CI workflows passing. The verifier
checks the SHA and a clean checkout, but it cannot independently authenticate
GitHub CI from an offline host. The model file must be under `--models-root`,
must match `AI_TRADING_MODEL_PATH`, and must match the expected SHA-256. A
missing model identity blocks the preflight, including in paper mode. The
pre-migration phase requires a readable existing Alembic revision in the
checkout's migration history and a matching expected head; it permits an older
applied revision on that path. The final phase
requires the applied revision to equal both the specification and the head. A
missing SQLite database is never created by the check. PostgreSQL credentials
are read from the managed runtime environment and are never written to the report.

A passing preflight is one deployment prerequisite. The staged packaged unit
syncs the runtime environment once, verifies identity before migration, runs
Alembic, then verifies identity again before starting the service. It does not
resync `.env` between the check and start. Installing this unit without a valid
release specification will prevent startup. It does not prove that a
model passes freshness, replay or promotion gates, that broker exposure is
safe, or that rollback across a schema change works. Before deployment, review
positions and open orders, verify CI and model-governance evidence, perform an
isolated migration-aware rollback rehearsal, and obtain any required live
approval. A clean CI-tested release specification, unit installation and
runtime verification remain outstanding.

The isolated SQLite regression rehearsal starts from the prior OMS revision,
preserves two duplicate-sequence events in a database snapshot, upgrades to
the checkout head, verifies deduplication, then restores the snapshot and
verifies the original revision, event sequences and database integrity. It
does not exercise PostgreSQL rollback or prove recovery of a live broker state;
those checks remain required for any different deployed database or migration.
