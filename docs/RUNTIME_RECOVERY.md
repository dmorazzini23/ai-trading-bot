# Runtime recovery boundary and rehearsal

**Status (2026-10-01 UTC):** the installed daily backup timer is enabled/active;
its September 30 23:30:23 UTC run succeeded. Current-version restore and a
controlled actual noncurrent-version restore have passed. The local real
PostgreSQL owner-crash/takeover drill passed separately. Cross-host failover,
funded-live accounting and unavailable original historical versions remain
unproven. No live activation, broker order or trading-service restart was part
of the latest recovery drill. Earlier dated observations below are historical.

## Controlled version restore (October 1 UTC)

Two uploads of the same previously verified September 29 bundle under a new
key in the approved `pruned/recovery_backups/` prefix produced distinct versions.
The first, now-noncurrent version was retrieved with `GetObjectVersion`; its
version ID, phase metadata and SHA-256 matched. Original objects were untouched;
the approved 30-day retention applies to the fixture. Isolated restoration
recovered 3,045 entries in 11.813 seconds; both SQLite integrity checks passed.
Fresh paper broker evidence at 04:44:58 UTC matched the restored account and
flat position boundary, with zero active orders. The archived code identity
remains historical `edfefa72f`, and restored order authority remains disabled.
No replacement owner was started. This does not establish full cash/fees/history
reconciliation or recovery of absent original versions. Evidence is recorded at
`/tmp/five-items-version-recovery-20261001T043836Z/proof.json` and
`broker-boundary-proof.json` in that directory. See
`TRADING_IMPROVEMENTS_20260930.md` for the actual PostgreSQL drill and remaining
cross-host acceptance criteria.

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

The packaged backup-sync unit creates and verifies a bundle before invoking
the optional uploader. The staged timer is configured for 23:30 UTC daily,
after the regular US equity session in either daylight-saving season. Local
retention is seven days and at most 14 snapshots. The uploader now selects
only the newest regular recovery bundle, verifies it, requires a 12-digit
`AI_TRADING_BACKUP_S3_EXPECTED_BUCKET_OWNER`, uploads with SSE-S3 and a SHA-256
checksum, then downloads the current object and compares its bytes. It does
not upload other runtime archives or delete remote objects. Remote retention
depends on the reviewed bucket lifecycle, not uploader settings.

On September 23 a read-only `ListObjectsV2` on the configured backup prefix
succeeded with zero keys, and the bucket lifecycle policy had an enabled
30-day expiration rule covering that prefix. `GetBucketVersioning` returned
`AccessDenied`, but the later owner-approved single-bundle upload returned a
version ID. The current object was downloaded, hash-checked and restored in
isolation. At that time, a version-pinned read was denied `s3:GetObjectVersion`
and the timer was disabled. Subsequent owner-approved IAM, retention and unit
installation enabled the daily timer and verified its real upload/read-back.
The October 1 controlled drill above supersedes the earlier version-read gap;
it does not reconstruct versions that never existed for historical unique keys.

A failed local backup exits nonzero, writes
`runtime/recovery_backup_latest.json` with a safe reason
class, and leaves no published partial bundle. A failed optional S3 sync fails
the systemd unit and must be handled separately; local success is not proof of
off-host durability. The service/timer changes require deployment and a
non-sending incident check before unattended coverage can be claimed.

When S3 sync is enabled, missing or corrupt recovery bundles, missing or
invalid bucket-owner configuration, upload failure and read-back mismatch all
fail the unit. A successful byte comparison proves that the current object was
readable at that time; it does not prove future retention or version recovery.
Inspect the unit journal for the safe failure class. Rehearse an isolated
restore from a scheduled remote object before counting the timer as off-host
recovery evidence.

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
   the September 30 follow-through also passed the actual PostgreSQL advisory-lock
   contention/crash/takeover test using a temporary localhost PostgreSQL 16.15
   server unpacked under `/tmp` without a system installation. The server was
   stopped afterward. This is local process-boundary evidence; real cross-host
   failover remains unverified. Do not treat either test as cross-host proof.
   An opt-in real process-boundary drill now lives at
   `tests/integration/test_postgres_owner_recovery.py`. It requires the managed
   `AI_TRADING_TEST_OWNER_DATABASE_URL` setting pointing to a **local disposable**
   `ai_trading_owner_test` database distinct from the runtime database. It creates
   no schema, proves contention, kills the owner and verifies takeover. An
   unconfigured test is an explicit skip, not proof; a passing local test does
   not replace a cross-host stop/revoke/fence rehearsal.
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
reconcile all post-snapshot broker activity before new risk. Off-host S3 retrieval
and isolated file restore now have evidence; replacement-host provisioning and
stop/revoke/fence recovery still require a separate rehearsal. Independent
host-failure alert delivery was acknowledged in the September 30 AWS drill;
this does not prove every backup failure alert path.

## Schema-2 read-only-source rehearsal (September 23, 09:02 UTC)

With the deployed paper runtime environment loaded, the new backup tool read
the two configured SQLite databases and copied runtime/model files into an
isolated `/tmp/goal-recovery-schema2-20260923` bundle. No production status
file was written. The tool code was local and unpushed, while the run manifest
describes the still-running service; this bundle is rehearsal evidence, not an
approved operational restore point. Bundle creation and built-in verification
took 46.58 seconds;
verification and isolated restore took 12.45 seconds. The 105.43 MiB bundle
contains 3,042 members, including 3,028 model files. The manifest names
`oms_intents_paper_monday.db` and `pretrade_rate_limiter.db` as required; both
restored databases passed `PRAGMA integrity_check`. The currently configured
`.pkl` model artifact is included. The OMS snapshot contains
2,135 terminal intents and no nonterminal intents, at revision
`20260506_0001`. A separate simulator regression still covers restoration of
an unresolved accepted order and identity-based recovery without a second
submission. The contemporaneous running-service health check showed fresh
broker state with zero positions/open orders and the existing
`required_model_stale` readiness failure. This does not prove completeness of
broker activity after the snapshot, off-host recovery, real PostgreSQL owner
fencing, actual alert delivery or a bounded recovery-time objective. The timer
remains disabled.
