# Runtime recovery boundary and rehearsal

**Status (2026-09-23):** local backup and isolated restore implemented; the
packaged timer has not been installed/enabled for this change. No service
restart, live activation, or broker order was part of this work.

## Authoritative state and coverage

The deployed paper service uses `/var/lib/ai-trading-bot/runtime` and
`/var/lib/ai-trading-bot/models`. The installed
`ai-trading-runtime-backup-sync.timer` was disabled when inspected on September
23. Its former script only uploaded existing `*.bak.*.gz` archives; it did not
make a transaction-consistent OMS snapshot. The runtime held two SQLite
databases: the OMS intent/event store and the pretrade rate limiter. Archived
JSONL logs alone cannot restore either database.

`python -m ai_trading.tools.runtime_recovery_backup` now uses SQLite's online
backup API for each runtime `*.db` and checks each snapshot with
`PRAGMA integrity_check`. It includes named order, fill, decision, TCA, OMS,
and risk-state evidence files; the run manifest, release specification,
pre/post migration identity reports, effective policy and selected runtime
metadata; and regular model artifacts/JSON metadata under `models`.
The bundle manifest records per-file size and SHA-256, creation time, and the
run manifest's code/config/policy identities. Verification rejects missing,
extra, altered, or unsafe members and checks restored SQLite integrity.
For new schema-2 bundles, creation also resolves the configured OMS SQLite URL
and persistent pretrade limiter path and requires those exact databases inside
the runtime directory. Verification checks their names against the manifest's
SQLite members. A missing configured database, an out-of-scope database or a
non-SQLite authoritative store fails the backup and writes failure status;
it cannot produce a misleading success bundle. Older schema-1 bundles remain
readable, but their manifests do not certify configured-database completeness.

Environment files, known credential/private-key paths, and arbitrary runtime
files are excluded, including model files under sensitive directory names.
File-content screening is not a proof that a mislabeled model artifact contains
no secret, so inspect new model sources before adding them to recovery scope.
The archive is mode 0600 in a mode 0700 backup directory. Secrets
must be restored from the approved external source; a bundle does not supply
credentials or trading approval. Files whose size or modification time changes
while copied cause backup failure. The SQLite databases are each
transaction-consistent, but the two
databases and evidence files are captured sequentially, so the bundle is not
one atomic cross-file instant. The broker remains the authority for actual
orders, fills, cash and positions.

The packaged backup-sync unit now creates and verifies the bundle before
invoking the existing optional S3 archive sync. The timer is configured for
23:30 UTC daily, after the regular US equity session in either daylight-saving
season. Its retention is seven days and at most 14 local snapshots; the
existing S3 sync has separate configured retention. A failed local backup
exits nonzero, writes `runtime/recovery_backup_latest.json` with a safe reason
class, and leaves no published partial bundle. A failed optional S3 sync fails
the systemd unit and must be handled separately; local success is not proof of
off-host durability. The service/timer changes require deployment and a
non-sending incident check before unattended coverage can be claimed.

## Isolated restore procedure

1. Keep the replacement host's order-submitting service disabled. Confirm the
   old host is stopped or fenced, and revoke/rotate its broker credentials
   before granting credentials to a replacement. A local file lock alone cannot
   fence two hosts. Never start two order-submitting owners.
   The live OMS path now also requires a PostgreSQL session advisory lock on
   its authoritative database. A second process using that same database
   cannot acquire submit ownership; a lost owner session blocks new canonical
   broker submits. This is a second fence, not permission to skip stopping or
   revoking the old host. Two hosts pointed at different databases would not
   share the lock. The current paper SQLite deployment has no cross-host
   database fence. An isolated shared-lock simulator checks two candidate
   owners, contention, release, backend-lock loss and process-ID mismatch;
   no local PostgreSQL server is installed, so real PostgreSQL contention and
   failover remain unverified. Do not treat the simulator as cross-host proof.
2. Fetch a complete bundle into a restricted directory. Verify it with
   `./venv/bin/python -m ai_trading.tools.runtime_recovery_backup --verify BUNDLE`.
   Inspect the manifest's code/config/policy identities and obtain the matching
   tested release, schema migration and external secrets separately.
3. Restore only to a **new isolated directory** with
   `./venv/bin/python -m ai_trading.tools.runtime_recovery_backup --verify BUNDLE --restore-to NEW_DIRECTORY`.
   The tool refuses an existing target. Do not replace the running runtime
   directory in place.
4. With a deterministic broker simulator or read-only broker snapshot, compare
   every nonterminal restored intent by account, client order ID, broker order
   ID, status and cumulative fills. Leave unknown outcomes unresolved and block
   new exposure. Reconcile cash, positions and open orders independently.
5. Validate migrations, model/replay/promotion gates, effective configuration,
   health and a non-sending incident snapshot. Operator approval and the
   separately governed live-activation process remain required for live funds.
   Start only the single authorized owner after market close and broker
   exposure review; preserve safe risk reduction.

Rollback is to stop the replacement owner, restore the previously verified
bundle into a new isolated directory, reconcile against broker truth again,
and restart only after schema compatibility and all existing gates pass. Do not
roll back a database schema by copying files over an active process. If the
code cannot read a newer schema, use a migration-aware forward repair or keep
the owner stopped. Never flatten positions to make a restore or rollback easy.

## September 23 rehearsal and limits

An isolated read-only-source rehearsal copied the deployed runtime/model state
to `/tmp`, with no production restart: 43.73 seconds to create/verify a 106 MiB
bundle and 12.09 seconds to verify/restore. The two restored SQLite databases
passed integrity checks; the restored OMS database had 2,135 intents and the
bundle included 3,028 model files. Synthetic regression fixtures restored a
`SUBMITTING` intent, matched it to simulated broker acceptance, and prevented a
second submit. Process-fault fixtures separately cover accepted-but-unacknowledged
orders, partial fills, duplicate evidence, and a fill/cancel race.

These measurements cover file restoration only. They exclude host provisioning,
secrets, broker reconciliation, migration decisions and service warmup, so they
are not a full recovery-time objective. At a successful daily cadence, the
recoverable local snapshot can lag current broker state by nearly 24 hours,
plus snapshot/sync time; missed or failed runs have no bounded loss window.
Evidence files may have per-file timestamp skew. The restored owner must
reconcile all post-snapshot broker activity before new risk. Off-host restore
and actual alert delivery remain unproven while the timer is disabled.
