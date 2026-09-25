# Current handoff

## September 25 verified position gate

See `docs/VERIFIED_POSITION_GATE_20260925.md`. The runtime go/no-go position
check now fails closed without a recent raw broker activity/position bundle;
unbounded local history remains a separate diagnostic. The bundle is rebuilt
and checked for same-account identity, complete pagination, execution IDs,
position-changing activities, matching quantities, a 60-second closing age,
and agreement with a later current broker position read. A read-only 02:48 UTC
paper capture matched flat-to-flat with zero executions and showed the old
AAPL -3 / AMZN +1 discrepancy only in the diagnostic. Changed-file Ruff,
mypy, compile, and 124 mapped tests passed; 293 runtime-control/after-hours
integration tests passed. The validator's sandboxed localhost curl was denied;
host HTTP 503 was expected for `required_model_stale`, with active service,
fresh zero-exposure broker state and NRestarts=0. Exact-tip CI and installed
release identity remain the deployment gates. Model/replay gates, research
budgets, holdout, and AWS MFA recovery hold remain unchanged.

The first exact-tip CI run for `5cc70e826` (`36090302137`) failed on three
older report tests that expected local-history comparisons in the gated
`open_position_reconciliation` field. The gate correctly requires a verified
broker bundle; local-history comparisons now live in
`diagnostic_open_position_reconciliation`. Those tests were updated, and text
rendering now states the unavailable gate reason while showing the diagnostic
mismatch count separately. The 97 focused report tests passed. Changed-file
validation passed Ruff, mypy, compile and 16 mapped tests; its sandboxed curl
could not reach localhost. Host smoke found the active paper service with
NRestarts=0 and expected HTTP 503 solely for `required_model_stale`.
Deployment remains on hold pending the follow-up commit, exact-tip CI success,
fresh broker exposure review, and release identity checks.

## September 24 after-close gate and performance review

See `docs/GATE_AND_PERFORMANCE_REVIEW_20260924.md` for evidence and acceptance.
The runtime go/no-go reconciliation compares unbounded local trade history to
current broker positions; AAPL -3 / AMZN +1 is the documented pre-anchor
legacy difference. The independent September 24 session audit was flat to
flat with zero fills, and the broker activity capture had no September 23/24
activities. No new current-session position discrepancy was established.
The report now labels its reconstructed inventory diagnostic only and states
that no verified broker ledger was applied; the numerical mismatch and gate
block remain unchanged. Focused report/lineage tests: 75 passed. The first
mapped validation encountered sandbox-denied bytecode writes under `.codex` in
two audit tests. With a temporary `PYTHONPYCACHEPREFIX`, Ruff, mypy, compile,
and all 792 mapped tests passed; the validator's sandboxed localhost curl was
denied. A host health check passed its expected HTTP 503 response with only
`required_model_stale` as readiness failure; service active, NRestarts=0.
A read-only run on the actual report inputs retained both AAPL/AMZN
mismatches and showed the diagnostic-only label.

September 24 produced zero broker orders and fills. Net costs remain
unverified; the existing cost comparison has one slippage pair and zero net
cost pairs. Replay was 186/250 samples at -1.492 bps candidate net edge, and
`required_model_stale` remained. The consumed September 21 hypothesis stays
retired, holdout untouched, and no new experiment or gate change occurred.
Complete exact-tip CI and installed release-identity checks before treating
this report-metadata change as deployed.

## September 24 exact-tip validation and release identity

The first backup-uploader CI run `35953054771` failed only because the runner
does not create the repository `venv`; four uploader tests could not start the
bundle verifier. The script now uses the repository interpreter when present
and the runner's `python3` otherwise. All 13 focused uploader tests and the
changed-file validation (62 mapped script tests) passed locally. Commit
`d7bf43cba0b11547bca7672046634ce72605816d` is on clean `main`; exact-tip
CI `35955032742` passed with 7,204 tests, four skipped and 80.17% coverage.
CodeQL, Workflow Lint, SBOM, research, replay and determinism gates passed.

The hash-only installed release spec was updated to that tested commit, with
the old spec retained as a local rollback copy. Installed pre-migration and
normal identity preflights passed. The paper service remained active with
NRestarts=0; no further restart was needed. At 04:37 UTC, the paper broker was
closed and active with zero positions and open orders. Health HTTP 503 still
reported `required_model_stale`, and the backup timer remained disabled. The
AWS MFA case, versioned S3 restore permission, recurring-upload approval and
backup-unit installation remain open. No S3 transfer or trading change was
made for this CI repair.

## September 24 AWS recovery hold and paper service guard activation

The owner opened an AWS account MFA recovery case; IAM
`s3:GetObjectVersion` remains unavailable. No IAM change or S3 write occurred.
The owner installed the reviewed `ai-trading.service` unit. It byte-matched
the packaged file and systemd reported `NeedDaemonReload=no`. Both release
identity preflights passed against the clean CI-tested checkout and installed
release spec. Before the 03:39 UTC after-close restart, the paper broker was
active and closed with zero positions and open orders; a fresh local recovery
bundle `recovery.bak.20260924T033758Z-d33e3419.gz` verified successfully.
The restart succeeded with NRestarts=0, and both automatic identity reports
passed. Postrestart broker exposure remained zero. Health HTTP 503 still
reported only `required_model_stale`; a non-sending incident snapshot flagged
`health_degraded` but sent no alert. No live activation or strategy change.

The local backup uploader now selects one newest regular recovery bundle,
verifies it before transfer, requires the expected 12-digit bucket owner,
uploads with SSE-S3/SHA-256, reads back the current object and compares bytes.
It no longer uploads legacy archives or deletes remote objects. Thirteen
isolated uploader tests and 75 changed-file mapped tests passed, along with
Ruff, mypy, compile, Bash syntax and diff checks. This uploader change has
not made a real S3 call. The backup timer is still disabled; the installed
backup unit is the older variant. Next: exact-tip CI and release-spec update,
then owner approval of the exact recurring upload scope, required bucket-owner
configuration, host installation of the backup unit/timer, and a scheduled
read-back/isolated restore. Version-pinned restore still needs the AWS case
resolved and `s3:GetObjectVersion`. Live-risk/equity, model, monitoring and
accounting evidence gates remain unchanged.

## September 23 approved off-host restore and documentation tip

The owner approved one postdeployment bundle upload and the documentation
commit `be3aeaebf`. Exact-tip CI `35931333370` passed with 7,202 tests,
four skipped and 80.17% coverage; research, replay, determinism, CodeQL,
Workflow Lint and SBOM passed. The clean checkout is on `be3aeaebf`.
The hash-only runtime release spec was updated after CI and its installed
identity preflight passed. The running paper service was not restarted:
active, NRestarts=0, health HTTP 503 solely for `required_model_stale`.

Only the verified postdeployment recovery bundle
`recovery.bak.20260923T202932Z-548c7e87.gz` was uploaded to the configured
same-account S3 bucket with expected-owner and AES256. The 111,191,546-byte
current object downloaded with the same version ID returned by upload; its
SHA-256 matched the local bundle (`fd5de33bf744099d089f074bce26f99d6c8b7b8a51dd3b45a1638ff7a858f6dc`).
An isolated restore passed bundle verification and both SQLite integrity
checks; the restored OMS has 2,135 intents, 764 fills and 80,184 events.
Version-pinned download was denied by IAM `s3:GetObjectVersion`, so restore
from an older overwritten version is unproven. No broad sync, timer enable,
S3 deletion, actual alert, broker order or live activation occurred. Scheduled
off-host backup needs a reviewed unit installation and separate authorization
for recurring upload scope; live readiness remains blocked by the risk/equity,
model, monitoring and accounting evidence gaps below.

## September 23 after-close paper release (`5031ca47e`, deployed)

Exact-tip CI `35913045555` passed: 7,202 tests passed, four skipped, 80.17%
coverage; research, offline replay, determinism, CodeQL, Workflow Lint and
SBOM passed. The paper broker clock was closed with zero positions and zero
open orders. The clean-checkout release identity passed before and after the
20:27:27 UTC service restart against the reviewed code, sanitized config,
launch profile, schema `20260506_0001` and model SHA. The current service is
active with NRestarts=0, fresh broker state and zero exposure; HTTP 503 reports
only `required_model_stale`. A non-sending incident snapshot returned
`blocked_qualification`, `should_alert=false`; bounded startup logs contained
no structured ERROR/CRITICAL or traceback. No live activation or strategy
change occurred.

Pre- and postdeployment local recovery bundles were created, verified and
restored in new isolated directories. The postdeployment restore includes the
release spec and model; source and restore each have 2,135 intents, 764 fills,
80,184 OMS events and zero nonterminal intents. Both SQLite integrity checks
passed. At release time, this did not prove off-host recovery or broker
activity after a future snapshot; the subsequent single-bundle readback is
recorded above.

The installed service unit still lacks the staged release identity guards, and
the installed backup unit is the older archive-only variant with its timer
disabled. Host administrator installation of the reviewed packaged units,
daemon reload and timer enablement remain required. Do not claim scheduled
backup or automatic identity enforcement from this manual release. The selected
3% peak drawdown and 1% daily loss contract remains unenforced pending verified
same-account equity/cash-flow evidence and live-path wiring; live openings
remain blocked. Verified fees, legacy history, external monitoring and model
qualification remain separate evidence gaps. The `be3aeaebf` handoff commit
was documentation-only after the deployed `5031ca47e` code release; its CI
and subsequent release-spec identity check are recorded above.

## September 23 owner decisions and CI repair

Owner-authorized push of `c4c5d3d75` succeeded. CI `35903372692` reduced the
failure to the config-cache test: 7,201 passed, one failed, four skipped,
80.15% coverage. CodeQL, Workflow Lint, SBOM, research, replay and seed jobs
passed. Reproduction showed that `import ai_trading.config as config_pkg` can
resolve a stale parent attribute after package reimport; management then
invalidated another module instance. The public `config.reload_env` now clears
its own cache after management reload. The adverse alias test failed before and
passed after the fix. Changed-file validation passed Ruff, mypy, compile,
85 mapped tests, live read-only health and a non-sending incident snapshot.
Health remains degraded solely for `required_model_stale`, with fresh broker
state and zero positions/open orders. The repair is local pending a new
reviewed push and exact-tip CI; no runtime restart or deployment occurred.

CI `35888651186` later failed on three tests with 7,199 passed, four skipped,
and 80.15% coverage. The two retry tests inherited CI's `paper_observe`
profile, whose zero order budget blocked submission before retry; that failure
was reproduced locally. The config-cache test held a stale package reference
after another test reimported `ai_trading.config`. The test fixture now selects
`paper_trade` explicitly and the config test uses the currently registered
package. The adverse-profile retry reproduction and the focused config group
pass; `agent_validate_changed.sh --market-hours` passed Ruff, mypy, compile and
11 mapped tests. No runtime or trading policy changed. The repaired local tip
has not been pushed or validated by CI, so deployment remains on hold.

The owner explicitly authorized pushing local main through `18ce1c10e` to the
public GitHub repository. The fast-forward push succeeded; GitHub CI run
`35888651186` failed for that exact SHA. CodeQL, Workflow Lint and SBOM passed,
as did the CI research, replay and determinism jobs. No deployment is authorized
by the failed check.
The owner also delegated selection of the rough 3% risk limit:
the proposed live-capital contract now uses 3% verified peak-to-trough equity
drawdown and a separate 1% verified daily equity loss limit. At $1,000/$2,000
initial capital those reference amounts are $30/$60 and $10/$20. This policy
is not enforced yet: same-account cash-flow-adjusted loss evidence, durable
baselines and canonical live wiring remain missing, so live openings stay
blocked. See `docs/SMALL_ACCOUNT_RISK_CONTRACT.md`. All model, replay, cost,
freshness, provenance, promotion and holdout gates remain unchanged.

## September 23 off-host recovery evidence limit (`18ce1c10e`, pushed, not deployed)

Read-only `GetBucketVersioning` on the configured S3 backup bucket returned
`AccessDenied` (HTTP 403) at 11:22 UTC. The backup-sync timer is still disabled
and inactive. The earlier complete prefix listing showed zero recovery objects;
the enabled 30-day lifecycle rule therefore does not establish an off-host
restore point. Verify versioning with an authorized identity, then after a
CI-approved backup deployment, read back a real bundle and restore it in
isolation. No AWS resource, timer or runtime state was changed.

The `local only` labels below describe the state when each milestone was
written. The commits through `18ce1c10e` are now pushed, but not deployed.

## September 23 live low-level quantity guard (`c273deecd`, local, not deployed)

The live `_submit_order_to_alpaca` claimed-intent check allowed a zero request
quantity (and a zero intent quantity) to reach the Alpaca SDK. A broker-stub
regression reproduced the zero request reaching `submit_order`. The guard now
requires both quantities to be strictly positive before broker submission;
valid positive claimed orders remain allowed. No strategy or owner threshold
changed. Changed-file Ruff, mypy, compile and 20 mapped tests passed. Read-only
health: service active with NRestarts=3, HTTP 503 solely for
`required_model_stale`, fresh broker with zero positions/open orders; the
non-sending incident snapshot stayed `blocked_qualification`. No order, alert,
restart or deployment occurred. CI, after-close exposure review and deployment
remain pending.

## September 23 paper uncertain-order follow-up (`155f357be`, local, not deployed)

The paper submit retry fix still let an accepted order with a lost response
reset its durable intent from `SUBMITTING` to `PENDING_SUBMIT`; the opening gate
then saw no unknown exposure. `execute_order` now retains `SUBMITTING` for
ambiguous paper exceptions, failed 5xx/no-result outcomes and an empty broker
response. Definite 400 rejections remain terminal. Parameterized live/paper,
direct-timeout and definite-rejection tests passed with the execution and
process-fault suites (65 tests plus the added direct-timeout test). Changed-file
Ruff, mypy, compile and 44 mapped tests passed. Live read-only health: active
service, HTTP 503 solely for `required_model_stale`, fresh broker and zero
positions/open orders; non-sending snapshot `blocked_qualification`. CI,
after-close release review and deployed
session evidence are still pending. Do not restart or deploy during market
hours, and do not infer any improvement in trade expectancy from this safety
correction.

A follow-up (`63a203c99`, local only) uses the real SQLite `IntentStore` and `OrderManager` with
a simulated paper submit timeout. It confirms one submit claim remains
`SUBMITTING`, blocks new openings and still permits closing. The repository
changed-file validator passed Ruff, mypy, compile and 43 tests for this test-only
change; no broker request or runtime mutation occurred.

## Current decision and next gate (September 23)

GitHub `main` contains `c4c5d3d75`; CI failed on that SHA, and the local
config-cache repair awaits a reviewed public push and exact-tip CI.
Before the push, `make secret-scan` passed on tracked files and an added-line
scan across all 44 outgoing commits found zero likely live credential
assignments. These checks do not replace a completed CI run.

The local paper service remains deployed at the earlier release; nothing in
these commits and nothing in this goal has been deployed. The paper service is
active with NRestarts=3, fresh broker state and zero positions/open orders;
readiness remains blocked by `required_model_stale`. Preserve the September
9–December 8 holdout and all consumed research budgets. No strategy trial or
live activation is authorized.
Deployment remains on hold until CI, after-close exposure
review, release identity and rollback checks pass.

September 23 no-blind-retry correction (`f4344a294`, local only): a paper
submit timeout or native 504 was retried up to three times after broker lookup
failed, and the outer `bot_engine.submit_order` facade added another API-error
retry layer. New tests reproduced three attempts for one ambiguous result.
The engine now stops paper as well as live retries on ambiguous broker submit
outcomes; the outer facade submits once through the engine. Definite 429
rejections retain the engine's bounded retry with the same client order ID.
Changed-file validation passed Ruff, mypy, compile and 23 mapped tests;
48 execution/runtime tests and 23 process-fault/error tests passed separately.
Read-only host smoke: service active, HTTP 503 for the existing
`required_model_stale` gate, broker fresh with zero positions/open orders;
non-sending snapshot `blocked_qualification`/`health_degraded`. No broker
order, restart, alert or deployment occurred. Roll back the local fix by
reverting `f4344a294`; CI and postdeployment session evidence remain required.

| Goal area | Current status | Acceptance still missing |
| --- | --- | --- |
| Durable risk | Numeric policy selected; implementation blocked by source evidence | Verified same-account cash-flow-adjusted equity, durable baselines, 3% drawdown and 1% daily enforcement; live-path proof after CI/deployment. |
| Crash recovery | Implemented and verified in isolated simulator; awaiting deployment/session evidence | CI, runtime adoption and naturally occurring uncertain-order reconciliation without duplicate submission. |
| Disaster recovery | Local bundle and isolated restore verified; awaiting deployment and shared-store evidence | Scheduled backup/restore, off-host durability, real PostgreSQL two-owner drill and broker reconciliation on restored state. |
| Supported path | Code-path guards/profile identity/extraction implemented; awaiting deployment | CI, reviewed release spec and postdeployment checks; live replacement child-OMS contract remains held. |
| Unattended operation | Same-host connector and local alert state implemented; external dependency remains | Off-host failure detection, approved real delivery, operator receipt/acknowledgement and recovery drill. |
| Accounting and evaluation | Synthetic account-size mechanics and cash audit implemented; fee/position evidence blocked | Precisely timed corporate actions, position boundaries and verified per-execution fees; no verified net strategy P&L. Retire the consumed negative replacement-model hypothesis. |

Read-only September 23 10:45 UTC fee follow-up: the configured managed-secrets
provider hydrated the existing paper credentials in process memory. A
pagination-complete activity read from September 22 00:00 UTC returned eight
broker fills, no fee activity rows, and no fill fee amount field.
This does not prove zero fees or future posting completeness; verified net P&L
remains unavailable. No order, data purchase, or source-record rewrite occurred.
See `docs/ACCOUNTING_AND_RESEARCH_DECISION_20260923.md`.

Small-account evaluation follow-up (`ea30ee66d`, local only): `small_account_capacity`
previously treated a missing estimated fee as a zero cash reserve while still
returning feasible shares. A full-cash $1,000 fixture reproduced the optimistic
result. Missing fee assumptions now return zero feasible shares with
`estimated_fee_assumption_missing`; an explicitly supplied zero estimate remains
an estimate, never a verified fee. Case/whitespace variants of the same symbol
also silently overwrote position or pending-buy exposure; the evaluator now
rejects those ambiguous inputs. Seven focused tests and changed-file Ruff,
mypy and compile checks pass after the duplicate-symbol fix. This is offline
mechanical evaluation, not a
live risk gate or strategy trial. No runtime service changed. Run CI on this
exact tip and retain the separate broker
cash/equity and owner-limit acceptance gates.

September 23 pre-push validation: a full local run was interrupted after
multiple failures and a two-minute hang; **the full suite did not pass**. The
focused triage fixed a genuine lazy Alpaca circular import and updated four
regression fixtures/expectations for current broker error, account sync,
missing-SDK, and staged unit behavior (`267dcdb0c`, local only). The changed-file
validator passed Ruff, mypy, compile and 61 mapped tests; six import tests and
11 focused HTTP/audit/socket tests also passed. The HTTP test hung inside even
a minimal `asyncio.to_thread` call in the restricted sandbox and passed
host-side. Ten audit/socket tests passed host-side with a temporary pycache.
Other cached failures include sandbox-denied sockets/DNS and one ignored
research scratch file scanned by a broad source test; these do not establish a
passing CI run. No trading or strategy behavior changed. Next: obtain explicit
approval for publishing the **final** commit range, run CI on that exact tip,
resolve any CI failure, then review broker exposure and release identity after
close before deployment. Keep deployment on hold meanwhile.

CI-path test safety follow-up (`8afe4c61e`, local, not deployed): two full-suite tests used
the real data-fetch path and could request Alpaca bars with CI's dummy
credentials. The datetime test now uses a local Alpaca-style client and checks
the timezone-aware arguments; the pretrade-health test supplies a local bar
frame and asserts two checked symbols. Six tests in those files passed, plus
changed-file Ruff, mypy and compile. No broker request or strategy trial ran.
The full suite still has not been certified on the current tip; run CI after
the public push is approved.

Focused process-fault and live-submit tests were rechecked September 23:
`22 passed` (`tests/integration/test_oms_process_faults.py` and
`tests/execution/test_live_submit_owner.py`). The configured paper Trading API
still has no observed complete per-fill total-fee evidence. Alpaca's separate
Broker API Activity SSE can link some one-to-one fees, but period-wide charges
have no parent execution; see `docs/ACCOUNTING_AND_RESEARCH_DECISION_20260923.md`.
No strategy, gate, order, deployment or notification changed in this check.

Read-only infrastructure check at 09:17 UTC September 23: the installed
`ai-trading-connectors.timer` is enabled and active, with a completed invocation
at 09:17 and the next scheduled for 09:18. The runtime backup-sync timer is
disabled and inactive. The deployed runtime environment names an S3 bucket and
region, and the host AWS identity resolves, but `HeadBucket` returned HTTP 403
and CloudWatch `DescribeAlarms` returned AccessDenied. The 403 does not prove
that backup uploads are forbidden or that the bucket is absent; it does prevent
this identity from validating off-host backup coverage through that read.
CloudWatch alarm coverage also could not be inspected with the current identity.
Follow-up at 10:48 UTC: `ListObjectsV2` succeeded for the configured backup
prefix and returned zero keys with no truncation. The bucket has an enabled
30-day lifecycle expiration rule covering that prefix; bucket versioning was
not verified. `HeadBucket` 403 therefore did not prevent this narrower read.
No off-host recovery bundle is currently visible at the configured prefix.
After CI and release review, enable/test the backup unit, then read the remote
bundle back and restore it in isolation before claiming off-host recovery.
CloudWatch alarm coverage still needs authorized read evidence. No AWS resource
was created or modified in these checks.

Rollback for uninstalled commits is a scoped revert. For deployed code, use the
reviewed release identity and isolated migration-aware restore process in
`docs/RELEASE_IDENTITY.md` and `docs/RUNTIME_RECOVERY.md`; never overwrite an
active OMS database or flatten exposure to make rollback convenient.

## September 23 S3 backup handoff correction (`d86d5f721` local main, not deployed)

The new recovery bundle already matches the uploader's `*.bak.*.gz` pattern and
lives under its recursive runtime scan. A separate defect remained: with S3
sync enabled, no staged archive made the script exit successfully; legacy-only
archives could also appear to satisfy a sync while no authoritative recovery
bundle was copied. The uploader now requires a `recovery_backups/recovery.bak.*.gz`
member before calling AWS and returns nonzero when absent. Isolated fake-AWS
tests cover empty/legacy-only inputs and positive bundle selection; no real S3
write was made. Changed-file validation passed Ruff, mypy, compile, Bash syntax
and 65 targeted tests. A read-only host smoke returned HTTP 503 for the existing
`required_model_stale` gate; the non-sending incident snapshot reported
`blocked_qualification` with `health_degraded`. An S3 read-back and restore
remain required before claiming off-host recovery. Revert `d86d5f721` to roll
back the local change; do not enable the timer before CI and release review.
Follow-up `f8c9efaa8` fixes another uploader false success: S3 retention listing
errors and object deletion errors were discarded after a successful upload. The
script now returns nonzero with a safe failure class for either error. Two
fake-AWS tests first reproduced zero exit status and now pass; a successful
retention fixture also passes. Targeted validation passed Ruff, mypy, compile,
Bash syntax and 68 tests. An ad hoc snapshot without the deployed environment
initially probed the wrong port; repeating with packaged `:9001` reported
`blocked_qualification` and `health_degraded`, matching direct HTTP 503
`required_model_stale`. No S3 read/write, timer activation or notification was
performed. CI, reviewed bucket list/delete permissions, remote read-back and
isolated restore remain required. Roll back by reverting `f8c9efaa8`.
Follow-up `1e72dd425`: an enabled retention pass also returned success for
invalid days/delete-cap values and when the delete cap left older objects
behind. Five isolated cases reproduced that false success; the script now
returns nonzero for those cases. Eleven uploader tests and 73 targeted mapped
tests passed with Ruff, mypy, compile and Bash syntax. The deployed environment
has S3 sync enabled but uploader-side retention disabled; its 30-day setting
does not prove remote retention, and bucket lifecycle could not be inspected.
An ad hoc five-second health snapshot timed out while direct `/healthz` returned
HTTP 503 for `required_model_stale`; a repeat with a ten-second bounded probe
returned `blocked_qualification`/`health_degraded`. No live service or AWS
resource changed. Before enabling remote retention, review its policy and
permissions; before claiming off-host recovery, obtain a remote read-back and
isolated restore. Roll back this local correction by reverting `1e72dd425`.

## September 23 release identity preflight (`e455a40f8` local main, not deployed)

The read-only `ai_trading.tools.release_identity` preflight compares a reviewed
CI-tested commit, clean checkout, effective sanitized config/profile hashes,
applied OMS migration and checkout migration head, and the exact configured
model file/hash. It blocks missing or conflicting evidence, rejects extra spec
fields, and writes a report without credentials or database URLs. The staged
packaged unit checks identity before and after Alembic migration and syncs the
runtime environment only once. Recovery bundles now include the spec and both
identity reports. Six focused release tests plus recovery/manifest tests passed
(16 total); the changed-file validator passed Ruff, mypy, compile, six mapped
tests and `systemd-analyze verify`. The running unit was not replaced: the
service remained active with NRestarts=0, HTTP 503 for `required_model_stale`,
broker fresh with zero positions/open orders, and non-sending snapshot state
`blocked_qualification`. A CI-approved candidate, valid release spec and model
approval remain unavailable. An isolated SQLite migration rollback fixture
(`0a0be6030` local main) passed:
prior OMS revision and duplicate-sequence events were restored with
`PRAGMA integrity_check=ok` after an upgrade to head. This does not establish
rollback for a different deployed database engine or live broker state. Next:
clean tested candidate and reviewed spec, deployment-engine rollback rehearsal,
then CI/exposure review before after-close installation/restart. See
`docs/RELEASE_IDENTITY.md`. Roll back the local unit change by reverting its
commit; do not install the staged unit without the spec.
The pre-migration check (`5671c6277` local main) also requires the applied
revision to belong to the checkout's migration history; an unrelated revision
blocks startup. Sixteen
focused release/recovery/manifest tests passed after this change.

Risk-contract investigation: the current live-profile daily-loss function
accepts caller-provided `daily_loss_state` without account, timestamp or
cash-flow proof. It is not a verified equity-loss control for the intended
small live account. This is documented in `docs/SMALL_ACCOUNT_RISK_CONTRACT.md`
(`020b2ecae` local main);
no threshold or trading gate was relaxed. Next implementation needs a durable
same-account equity baseline and activity reconciliation, then a fail-closed
canonical evaluator and targeted live-path tests. The owner decision on 3%
meaning and daily limit remains pending.

Follow-up `443c8e429` local main (not deployed): canonical live openings discard
strategy-provided `daily_loss_state`/`loss_state` after refreshing broker
exposure, and the pure profile gate no longer reads a loss number from the
account payload. This deliberately blocks live openings until a verified
same-account, cash-flow-adjusted daily-loss source exists; paper behavior and
closing-order paths are unchanged. The active service then restarted three
times during Alpaca paper-account 504 timeouts: one native SDK `APIError`
escaped `_fetch_account_state`, and subsequent startup AUTO sizing refused to
guess capital after account timeouts. The local account-read patch catches the
native error with the bounded read helper, clears stale cached account data,
marks broker sync degraded with `failed_components=("account",)`, and blocks
new openings while preserving the closing phase allowance. Targeted tests
cover the 504, cached-account removal, opening block, and reduction allowance.
The running unit has not been replaced; broker health later recovered fresh
with zero positions/open orders, while readiness still reported
`required_model_stale`. Changed-file validation passed Ruff, mypy, compile and
338 mapped tests; 56 broker-sync/replay/degraded-gate checks also passed.
Final host-side read-only checks: active service, NRestarts=3 (unchanged after
07:54:51 UTC), HTTP 503 for `required_model_stale`, fresh broker with zero
positions/open orders, non-sending state `blocked_qualification`. No restart,
order, alert or deployment was performed by Codex. Next: CI and broker exposure
review before any after-close deployment; build a durable equity/activity
baseline before any live-capital proposal. Rollback is the scoped commit revert.
Provider evidence boundary (`5ded97ef7` local main): Alpaca's prior-close
`last_equity` and date-only
nontrade activities cannot by themselves certify an intraday cash-flow-adjusted
loss. The Trading API activity filter uses creation time and some charges post
on a later day. This is now explicit in the risk contract; missing activity
does not prove zero cash movement. The live loss gate remains blocked pending
better source evidence or an approved conservative operational procedure.

Broker contract review: Alpaca documents that `pending_new` orders cannot use
its native replace endpoint and that a successful replacement response does
not guarantee the old order was replaced. A safe cancel-and-resubmit therefore
needs a new canonical decision, child OMS identity and parent/child reconciliation
through fill/cancel races; the live helper remains held. See
`docs/SUPPORTED_TRADING_PATH.md` and the official Alpaca order docs.

Submission-path follow-up (`1653455b1` local main, not deployed): the real Alpaca/Tradier
adapter submit methods now reject configured live mode before the client call.
The short-cover and canonical low-level submit handlers now recognize the
SDK's native `APIError` in addition to timeout/connection failures. For an
ambiguous native 504, the live retry wrapper makes one attempt, retains the
claimed OMS intent as `SUBMITTING`, and suppresses broker failover while
identity-based reconciliation remains pending. Regression fixtures cover both
adapters and native-504 cover/ordinary submissions. These changes preserve
paper/test adapter behavior and do not relax live qualification or risk gates.
Changed-file validation passed Ruff, mypy, compile and 316 mapped tests;
314 focused adapter/execution/error tests and 30 replay/degraded-gate tests
passed. The final mapped run included plain-text native-504
regressions were added. Its sandboxed localhost smoke could not connect;
the separate host-side read-only check did reach the service. Host checks:
service active, NRestarts=3 (unchanged), HTTP 503
for `required_model_stale`, fresh broker with zero positions/open orders;
non-sending snapshot `blocked_qualification`. No order, alert, restart or
deployment was performed by Codex. Next: CI and an after-close deployment
review; the 3% meaning, daily loss limit and verified equity source remain
outstanding. Roll back by reverting the scoped local commit.

## September 23 local alert recovery and acknowledgement (`40ee9a8b7` local main, not deployed)

The Slack incident helper previously retained its last signature when triggers
cleared, so an identical future incident could be suppressed as a duplicate or
by the previous incident's cooldown.
It now records local resolution and preserves the prior incident while allowing
the next occurrence to alert. A signature-matched `acknowledge_incident` tool
records an operator locally without suppressing alerts or claiming notification
receipt. A fake-webhook regression covers first alert, duplicate suppression,
acknowledgement, silent recovery and recurrence. Changed-file validation passed
Ruff, mypy, compile and 49 mapped tests. A non-sending snapshot reported
`blocked_qualification`/`health_degraded`. No notification, service restart or
deployment occurred. Off-host detection and actual delivery/acknowledgement
remain external prerequisites for unattended live operation.
Follow-up (`7c92d872c` local main, not deployed): a single unconfirmed health outage previously left pending state
across a recovered probe or a deduplicated different incident. The next
separate outage could then skip its first confirmation. Pending state now clears
on either transition. Two isolated regressions and the full 51-test alert file
passed with Ruff, mypy and compile; the non-sending snapshot remained
`blocked_qualification`. No message or runtime action was sent.
State-write follow-up (`5d86f7a90` local main, not deployed): the connector, acknowledgement and
clear tools previously used unlocked read/modify/write on one incident JSON
file. A concurrent timer run could overwrite an operator's local
acknowledgement. They now share a same-host file lock and atomic fsynced
replacement. A forced-overlap fake-webhook test and the full 52-test alert file
passed with Ruff, mypy and compile. A non-sending snapshot remained
`blocked_qualification`; no notification or deployment occurred. This remains
local state, not verified external delivery or cross-host acknowledgement.

## September 23 pending-order cancellation boundary (`ca7ff61d6` local main, not deployed)

The live replacement helper introduced in `9f49d55ce` returns before its own
cancel call, but pending-order maintenance can separately cancel the original
when a replacement fails. That contradicted the earlier broad claim that the
original always remained. Pending-order maintenance now checks broker closing
`position_intent` or the matching durable OMS intent before automatic
cancellation and preserves a verified closing order; stale opening orders
retain the configured cancel policy. Fifteen targeted tests cover identity
mismatch, opening and closing regressions. Changed-file validation passed Ruff,
mypy, compile and 287 mapped tests; 28 replay/live gate tests passed. Host-side
read-only health returned HTTP 503 solely for `required_model_stale`, broker
fresh with zero positions/open orders, service active with NRestarts=0.
The non-sending incident snapshot reported `health_degraded`. No restart,
deployment or alert delivery occurred. The live replacement itself still needs
a durable pretrade and identity contract. Rollback is a local commit revert.

## September 23 low-level live submit boundary (`9f49d55ce` local main, not deployed)

`_submit_order_to_alpaca` now requires a claimed canonical OMS intent with the
same client ID, symbol, side and sufficient authorized quantity in live mode.
After an ambiguous live broker result, `_execute_with_retry` makes no blind
second submit; `execute_order` leaves the intent `SUBMITTING` instead of
resetting it to `PENDING_SUBMIT` or marking an unverified rejection. The live
cancel-and-resubmit limit replacement returns before canceling the original,
because its new client ID has no durable replacement claim. Paper behavior is
unchanged. Focused tests cover direct bypass, claimed-intent success, malformed
quantities, lost and empty responses, and retained original order. The separate
execution suite passed 705 tests after correcting a stale quote-policy test
fixture that lacked a fake OMS store. The final finite-quantity and intent-ID
guard then passed 45 focused tests; 28 replay/live gate tests passed. The
changed-file validator passed lint, type, compile and 58 collected tests with
`--market-hours --skip-runtime-smoke`; its default sandboxed localhost curl
failed. Host-side read-only health returned HTTP
503 for existing `required_model_stale`; broker was fresh with zero positions
and open orders, service active with NRestarts=0. The non-sending incident
snapshot completed and listed `edge_realism_gap_high`, `go_no_go_failed`,
`go_no_go_failed_checks` and `health_degraded`; no alert was sent. No service
restart or deployment occurred. The replacement path still needs a durable
pretrade design before live activation. Rollback is a local commit revert.

## September 23 direct-submit and short-cover follow-up (`fa6aaa380` local main, not deployed)

The generic `ExecutionEngine.safe_submit_order` now rejects live mode because it
does not carry canonical pretrade and OMS evidence. A live opposite-side short
cover now refreshes broker positions, open orders and account identity; rejects
stale, missing or conflicting position data and same-symbol pending orders or
intents; claims a durable OMS intent before submitting with its client order
identity; and records a verified broker acknowledgement. An ambiguous response
leaves the intent unresolved so a later attempt cannot blindly resubmit.
Paper behavior and strategy/model gates are unchanged. Focused tests cover
successful cover, stale/conflicting state, an unresolved intent and a lost
broker response. Changed-file validation passed Ruff, mypy, compile and 284
mapped tests; 28 replay/live gate tests also passed. Its sandboxed curl failed;
a separate host read-only health check returned HTTP 503 solely for existing
`required_model_stale`, with fresh broker, zero positions/orders and active
service (NRestarts=0). The non-sending incident snapshot reported
`health_degraded`. The lower-level `_submit_order_to_alpaca` boundary and
replacement callers still require pretrade/claim proof before live activation.
No running service was changed or restarted. The scoped code and docs are
committed on local main; continue the remaining direct-path proof. Rollback is
a local commit revert.

## September 23 live opening exposure evidence (`548fefd74` local main; awaiting deployment)

The live-canary/profile exposure evaluator previously skipped positions or
open orders with missing quantities/prices and used candidate price or cost
basis as a fallback. It also trusted a caller's closing label without checking
whether the requested quantity would flip the position. Live openings now fail
closed on missing account identity, position/open-order snapshots, unknown
current market value, unpriced pending orders, and account conflicts. A bounded
reduction must fit the observed position. The canonical live precheck refreshes
broker state and account for each opening, overwrites caller-supplied exposure
arrays, and blocks stale/failed reads. A broker snapshot update failure remains
stale, and the profile gate receives the engine's actual execution mode.

Forty targeted profile/risk tests passed. Final
`agent_validate_changed.sh --market-hours` passed Ruff, mypy, compile and 326
mapped tests; its sandboxed localhost curl failed. Another 28 targeted
replay/live gate tests passed. Separate host-side health reached the unchanged
paper service: HTTP 503 from existing `required_model_stale`/replay flags,
broker fresh with zero positions/open orders; active service and NRestarts=0.
The non-sending incident snapshot reported `blocked_qualification`.

This does not approve a 3% interpretation or daily-loss limit, implement a
durable high-water mark, or prove every direct broker submit/reduction path.
Next: close those paths with fresh same-account position/order evidence and
durable OMS state, then validate an isolated live-profile simulation. No live
activation, push or deployment occurred. Rollback is a local commit revert.

## September 23 operational-state triage (`48e7308f3` local main; awaiting deployment)

The separate connector timer was enabled and completed successfully at
06:11 UTC; the older healthcheck timer was disabled. Its existing incident
snapshot now labels healthy abstention, blocked qualification, degraded data,
uncertain broker state, execution incident, unavailable monitoring and unknown
degradation without changing `/healthz` readiness, alert triggers, delivery or
trading gates. The label is advisory and is included in any future incident
text. Forty-eight focused tests, Ruff, mypy and compile passed under
`agent_validate_changed.sh --market-hours`; the sandboxed localhost curl was
inaccessible. A separate host-side non-sending snapshot identified the current
paper service as `blocked_qualification`, with existing `health_degraded`
trigger, broker connected and `required_model_stale`. No notification was sent.

This same-host timer can detect application failure while the host stays up;
it cannot detect loss of the host or its own timer. Off-host heartbeat,
notification approval, acknowledgement and recovery verification remain
required for unattended live use. The commit is not pushed or deployed.
Rollback is a local commit revert; preserve the original incident snapshot and
readiness controls.

## September 23 account equity boundary capture (`5a93ef693` local main; awaiting session evidence)

`ai_trading.tools.broker_accounting_evidence.capture` now records the cash,
equity, currency and observation time returned by the existing read-only broker
account call. `reconcile_account_equity` and optional CLI account-boundary flags
audit cash effects from identified broker executions and precisely timed USD
cash activities between two same-account boundaries. Date-only charges, missing
amounts, identity conflicts and any cash difference remain unverified. The
report shows observed equity/position-value changes without granting verified
net-strategy performance or allocating account fees to fills. September 22
lacks the new boundaries, so its seven unknown verified fees remain unknown.
No broker order, experiment, strategy change or protected holdout evaluation ran.

Changed-file validation passed Ruff, mypy, compile and 10 mapped tests; the
sandboxed health curl was inaccessible. After the CLI regression was added,
12 focused tests plus file Ruff/mypy/compile passed. Separate read-only
health at about 06:02 UTC reached the unchanged paper service: HTTP 503 from
existing `required_model_stale`/replay flags, broker fresh with zero positions
and open orders. This commit is neither pushed nor deployed. After CI and a
safe after-close deployment, capture two future, time-aligned account
boundaries and covering activities; inspect any date-only fees and corporate events as
unresolved, then compare broker cash/equity with separate position and fee
evidence. No historical net P&L claim follows from this patch.
Corporate-action follow-up (`1e9eaa215` local main, not deployed): the cash audit now reports
observed cash distributions and position-changing corporate actions with their
timing relation to the boundary. A timestamped zero-cash split can still have
`cash_reconciled` status, but its position effect remains explicitly unverified;
a date-only split remains unplaced and makes the cash interval unverified.
Fourteen focused/mapped tests, Ruff, mypy and compile passed. This does not
resolve the absence of position boundaries, causal times for broker nontrade
rows, or per-execution total fees. No source record or trading gate changed.

## September 23 submit-owner fence (`2ea228e91` local main; awaiting PostgreSQL and deployment evidence)

Live `OrderManager` initialization now requires an exclusive PostgreSQL session
advisory lock on the authoritative OMS database. A second owner cannot start;
lost or forked owner sessions fail closed at the submit claim and immediately
before the canonical broker SDK call. Standalone `alpaca_api.submit_order` and
`bot_engine.safe_submit_order` reject live mode. Paper/diagnostic behavior and
all model, replay and research gates remain unchanged. No live order was sent.

Changed-file validation (`bash scripts/agent_validate_changed.sh --market-hours`)
passed Ruff, mypy, compile and 44 mapped tests; its sandboxed localhost curl
could not connect. Separate read-only health at 05:54 UTC reached the unchanged
paper service: HTTP 503 for existing `required_model_stale` and replay flags,
fresh broker with zero positions/open orders, active service and NRestarts=0.
The non-sending incident snapshot passed structurally and reported
`health_degraded`. Synthetic two-process owner/loss tests pass, but there is no
configured live PostgreSQL database to prove two real hosts, lock loss, or
failover. Before deployment, run CI, check broker exposure, and perform an
isolated same-database PostgreSQL standby/restore drill. Do not start live
trading on this evidence alone. Rollback is the local commit revert; preserve
the prior operational stop/revoke fence during recovery.

## September 23 phased trading-system hardening (in progress)

Goal: implement six areas in dependency order: risk-state integrity and an
approved small-account risk contract; isolated OMS crash recovery; demonstrated
backup/restore; supported runtime profile and path map; unattended operation and
deployment identity; and account-size accounting/evaluation. No live activation,
new strategy experiment, holdout use, or change to model/promotion gates is
authorized. The owner's 3% drawdown meaning and a separate daily-loss limit
were requested asynchronously; thresholds remain unchanged pending that answer.

First risk-state milestone, committed locally as `687e516ca`:
`ai_trading/runtime/live_canary.py`
now rejects malformed, unreadable, future-dated, and missing historical canary
state. A lock serializes enforced-profile evaluations. The event journal is
fsynced before the snapshot so an ambiguous write consumes budget
conservatively; same-day journal attempts are reconciled with the snapshot.
A persistent initialization marker distinguishes a fresh empty directory from
loss of both state and event files after initialization. New regression cases
cover corruption, missing history, journal-ahead restart, valid rollover,
future date, and concurrent attempts. `tests/runtime/test_live_canary.py`: 20
passed. `agent_validate_changed.sh --market-hours` passed lint, type, compile,
and 20 mapped tests; its sandboxed localhost curl failed. A separate read-only
health call reached
the active paper service: broker fresh with zero positions/open orders; readiness
503 remains due to `required_model_stale`, and replay parity remains flagged.
The service has not been restarted for this worktree change.

Second OMS milestone, committed locally as `d29bf1049`: stale `SUBMITTING` intents previously
became `FAILED` before broker lookup, and `claim_for_submit` could reclaim them
after a timeout. Both permitted a duplicate after an accepted-but-lost broker
response. Reconciliation now checks broker identity, links accepted/partial/
terminal responses, and leaves unknown outcomes open with a warning; claiming
is limited to `PENDING_SUBMIT`. Isolated process tests persist simulated broker
acceptance, terminate before local acknowledgement, then verify restart,
partial fill, fill/cancel race, and no duplicate claim. Focused OMS checks:
21 passed after updating an older test that expected the unsafe stale close.
The initial mapped validator had 718 passed/one failed at that old expectation;
the rerun passed lint, type, compile and 45 mapped tests. Its sandboxed health
curl failed; the separate read-only health result above remains the runtime
check. Process tests do not contact the running service or broker.

Follow-up OMS opening gate (`c63cda3cb` local main, not deployed): an unresolved
durable `SUBMITTING` intent now blocks new openings across symbols after a
process restart, even when the general execution-phase gate is disabled.
Read failures block openings; a missing intent store blocks live openings.
Orders classified as closing positions remain allowed by this gate; their
separate pretrade controls still apply. A process-crash regression proves the
opening gate remains closed while broker acceptance is unknown, then opens
after identity-based reconciliation. Focused tests: 7 passed. Changed-file
validation passed Ruff, mypy, compile and 280 mapped tests; the sandboxed
localhost curl was inaccessible. A separate read-only host health check at
05:08 UTC returned HTTP 503 from the unchanged paper service: broker fresh,
zero positions/open orders, `required_model_stale`; service active with
NRestarts=0. A non-sending incident snapshot reported `health_degraded`.
This change is neither deployed nor a model/trading-gate relaxation.
Commit `39d4bd879` adds a process test that injects a failed acknowledgement write after
simulated broker acceptance, reopens the intent store, and verifies no reclaim,
blocked openings and identity-based recovery without a second broker order.
Its two focused process checks plus Ruff and mypy passed; it does not simulate
a physical disk failure or interact with the running service.

Follow-up claim contract (`c8c374140` local main, not deployed):
`OrderManager.begin_external_order_lifecycle` previously ignored a `False`
result from the durable submit claim and returned an intent ID anyway. It now
returns no ID and leaves no new in-memory mapping when the claim is refused.
When a paper engine has a configured durable store, that failure blocks the
broker submit even if paper durability was otherwise optional; live remains
fail-closed. Fake and real-store regressions cover claim refusal, crash-persisted
`SUBMITTING`, and no second simulated broker attempt. Focused tests: 9 passed.
Changed-file validation passed Ruff, mypy, compile and 60 mapped tests; its
sandboxed curl could not connect. Separate host health at 05:33 UTC returned
HTTP 503 solely for `required_model_stale`, broker fresh with zero positions and
open orders; service active, NRestarts=0. Non-sending incident snapshot reported
`health_degraded`. This prevents reuse of one claimed intent, but does not fence
two owners creating *different* new intents on separate hosts; PostgreSQL or
another shared ownership mechanism and all direct submit paths still need
explicit verification before live or restored-host activation.

Backup inventory: the installed `ai-trading-runtime-backup-sync.timer` is
disabled. Its script only uploads archived `*.bak.*.gz` files; it does not
snapshot the active OMS SQLite database. The deployed runtime has an active OMS
database under `/var/lib/ai-trading-bot/runtime`. Consistent snapshot creation,
integrity/retention/failure checks, isolated restore and ownership fencing are
still required. No secrets were copied or printed.

These commits are on local main and have not been pushed or deployed. The
remaining risk contract, runtime-path, operations, and account-size evaluation
phases are incomplete. This work does not establish live readiness. Preserve
the pre-existing uncommitted handoff content and
`docs/REMAINING_TASKS_HANDOFF_20260922.md`.

### Recovery bundle milestone (`ddbacc1ac` local main, not deployed)

`ai_trading/tools/runtime_recovery_backup.py` creates online SQLite snapshots
of the two runtime databases, captures named runtime evidence and model files,
and verifies per-file hashes and SQLite integrity. It excludes environment
files and sensitive model filenames, writes mode-0600 bundles, records safe
success/failure status, and applies explicit local retention. The packaged
backup-sync unit now creates a bundle before optional S3 sync; its timer is
configured for 23:30 UTC daily. It remains disabled on the host.

Synthetic backup/restore tests: 3 passed, including restored OMS intent versus
simulated broker acceptance, retention and corruption rejection, and failure
status. Ruff, mypy, compile and mapped tests passed under
`agent_validate_changed.sh --market-hours`; `systemd-analyze verify` passed.
The validator's sandboxed curl could not connect. Separate local health at
04:50 UTC reached the active paper service: fresh broker, zero positions/open
orders, readiness degraded only by `required_model_stale`, replay parity still
flagged; NRestarts=0. Non-sending incident snapshot passed.

Latest isolated read-only-source rehearsal against deployed data: bundle
creation/verification 43.73 seconds, isolated verify/restore 12.09 seconds,
106 MiB archive, two integrity-checked restored SQLite databases, 2,135 OMS
intents and 3,028 model files. The archive and restored manifest exist under
`/tmp/goal-recovery-rehearsal-final` and `/tmp/goal-restored-final`. See
`docs/RUNTIME_RECOVERY.md` for exact scope, recovery procedure, data-loss
boundary and limitations. No production backup, service restart, broker order,
remote sync, or live activation occurred. Local main is six commits ahead of
origin/main. Next: finish risk-contract/submission-path verification and the
other four goal areas.

Backup completeness follow-up (`133b285f7` local main, not deployed): the bundle creator could
previously succeed if any runtime SQLite file existed, even with the
authoritative OMS or persistent rate-limiter database absent. New schema-2
bundles resolve the configured stores and require their identities at creation
and verification; non-SQLite or out-of-scope authoritative stores fail closed.
Sensitive model-directory components are excluded. The deployed paper
configuration resolves to the present `oms_intents_paper_monday.db` and
`pretrade_rate_limiter.db`; no production bundle was created in this follow-up.
Six isolated backup tests, Ruff, mypy and compile passed; the non-sending
snapshot remained `blocked_qualification`. Legacy schema-1 bundles still
verify integrity but cannot certify configured database completeness. The
backup timer remains disabled; next is CI, after-close exposure review and a
planned timer deployment plus an actual scheduled backup and restore check.
Schema-2 isolated rehearsal (`62f421a02` local main) at 09:02 UTC used the deployed paper runtime
environment and wrote only to `/tmp/goal-recovery-schema2-20260923`. A 105.43
MiB bundle with 3,042 members was created/verified in 46.58 seconds and
verified/restored in 12.45 seconds. Its required OMS and pretrade limiter
databases passed integrity checks; the restored OMS had 2,135 terminal intents,
none nonterminal, at revision `20260506_0001`. The configured `.pkl` model
artifact was included. Running broker health remained
fresh with zero positions/open orders; readiness still failed on
`required_model_stale`. This read-only-source rehearsal does not replace an
actual scheduled backup, off-host restore or real PostgreSQL owner drill.
Owner-fence follow-up (`42c4c17fa` local main, not deployed): no PostgreSQL binary, server unit,
or container runtime is available on this host. Two isolated shared-lock tests
exercise two candidate OMS owners, contention, release, lock loss and process-ID
mismatch; Ruff, mypy, compile and both tests passed. This verifies the local
fence protocol only. A real PostgreSQL two-owner/failover drill against one
shared database remains an explicit acceptance requirement before live or
restored-host submission.

Live-submit extraction (`b51897307` local main, not deployed): the oversized submit handler now
uses a pure ambiguity classifier for native broker 5xx, timeout/connection and
plain-text provider errors. Direct cases distinguish uncertain outcomes from
ordinary 400 rejections without changing identity lookup or failover policy.
The focused execution files passed 302 tests; the changed-file gate passed
Ruff, mypy, compile and 18 mapped tests; 39 replay/degraded-gate tests passed.
Read-only host health remained degraded for existing
`required_model_stale`/replay flags, with fresh broker state and zero positions
or open orders. No runtime restart, broker order or deployment occurred.

### Small-account mechanics (`10291462c` local main, not deployed)

`ai_trading/tools/small_account_capacity.py` evaluates explicit $1,000 and
$2,000 long-only whole-share scenarios with pending buy commitments, cash
reserve, minimum/maximum order notional, gross/symbol concentration and
per-side cost assumptions. It keeps estimated market costs, estimated fees and
unknown verified fees separate. Four synthetic tests pass, including both
capital sizes, concentration, pending cash, high-price whole-share failure,
invalid limits and fee provenance. Changed-file validation passed Ruff, mypy,
compile and four mapped tests; the sandboxed health curl was inaccessible, with
the separate health smoke above covering the service. No strategy trial or
holdout evaluation ran. See `docs/SMALL_ACCOUNT_EVALUATION.md` for scope and
retirement/observation/live-canary eligibility rules. Actual equity/cash
reconciliation and the owner-approved 3%/daily-loss thresholds remain open.

`docs/SMALL_ACCOUNT_RISK_CONTRACT.md` (`b77c3548f` local main) states the required account/equity,
daily/starting/peak-loss, outstanding-order, stale-broker and safe-reduction
semantics without selecting a numeric loss limit. The owner decision on the
meaning of 3% and, if needed, a separate daily limit is still pending. This
document is a contract draft; it is not wired into all submit paths and grants
no live authority. Docs-only validation passed.

### Runtime path and configuration identity (`07d0ca4d0` local main, not deployed)

The startup run manifest and order lineage used different JSON serialization
for the same sanitized `TradingConfig`, yielding different hashes. The
canonical hash is now shared. The sanitized snapshot includes a digest of all
declarative settings, with secret values represented only as configured/absent;
the manifest separately captures the resolved launch-profile payload and hash
for policy overrides outside the config schema. Tests verify sensitivity to an
otherwise omitted position-size setting, stability across secret-value changes,
manifest/order-hash agreement, and launch-profile override identity. The path,
precedence, units and known identity gaps are mapped in
`docs/SUPPORTED_TRADING_PATH.md`. Previous hashes have different scope and must
not be rewritten. This does not yet bind the applied schema revision, exact
loaded model artifact or checkout dirty state to a tested release.

Focused tests: 21 passed; four related decision/manifest cases also passed.
Changed-file validation passed Ruff, mypy, compile
and 155 mapped tests; its sandboxed health curl could not connect. The first
host health request timed out at 5 seconds; a bounded repeat returned HTTP 503
in 2.82 seconds, broker fresh with zero positions/open orders and only
`required_model_stale` as readiness failure. Service remained active with
NRestarts=0; a non-sending incident snapshot reported existing degraded
health and go/no-go flags. No restart, broker submission or deployment occurred.
Read-only host timer status on September 23: the packaged healthcheck,
runtime-backup-sync and runtime-report timers are all disabled. The health
runner can send a webhook if configured, so installing/enabling it requires
tested state classification and non-sending alert fixtures before notifications
are approved. The same-host timer cannot detect total host loss; off-host
heartbeat evidence remains an external dependency. No timer was enabled.

## September 23 accounting and research decision

Detailed evidence and acceptance criteria: `docs/ACCOUNTING_AND_RESEARCH_DECISION_20260923.md`.
September 22 has seven local fill rows, but eight broker executions: the MSFT
end-of-day sell-2 row combines two sell-1 broker activities. The AMZN and MSFT
end-of-day exits match same-account orders but have no strategy decision or
resolved TCA. Reconciliation now names those gaps without inventing matches;
broker accounting now flags execution-count mismatches even when order quantities
match. Future EOD submissions carry causal trigger time and intent reason in
order events. The original September 22 records remain unchanged.

No September 22 fill has a verified total fee. Captured broker fills have no
fee field; account-level fee rows have no execution reference and cannot be
allocated. Net P&L remains unknown. The broker-backed September 7–18 quantity
ledger and independent September 22 quantity interval may be reported separately;
pre-anchor legacy history remains excluded. AAPL -3 / AMZN +1 predates the
verified flat September 18 opening, with no supported exact origin.

Retire the consumed September 21 replacement hypothesis (negative development
economics). Replay remains blocked at 191/250 samples and negative candidate
net edge. Keep `required_model_stale` and every reset/holdout/promotion gate.
No new research trial was run or proposed.

Commit `3c206654c` is on local and remote main. CI run 35813717726 passed:
7,068 tests passed, four skipped, 80.07% coverage. CodeQL, SBOM, Workflow Lint,
deterministic replay, offline replay and research backtest gates also passed.
Locally, 73 focused tests passed; changed-file validator passed Ruff, mypy,
compile and 75 mapped tests. The validator's sandboxed health curl could not
connect, and a local full run stopped after 2,070 passes because the sandbox
forbids a free-port test's socket creation; CI passed in its normal environment.

At 03:45:38 UTC the direct paper broker check confirmed market closed, active
account, zero positions and zero open orders. The service was restarted after
close at 03:45:50 UTC. After warmup, health reports fresh broker connectivity,
zero positions/open orders, `paper_diagnostics_only`, and only
`required_model_stale` as a readiness failure; `replay_live_parity_gate_failed`
remains an attention flag. Service active, NRestarts=0. Two yfinance ERROR rows
reported an SPY daily backup download failure during startup; minute Alpaca IEX
health then returned 391 rows and completed. No execution or readiness error
was established. Monitor the next regular session and after-close audit without
forcing trades or changing gates. The earlier uncommitted handoff content and
`docs/REMAINING_TASKS_HANDOFF_20260922.md` were preserved outside the commit.

## September 22 CI repair and after-close deployment completed

Published 43abd3d0f after removing the NumPy substitute leaked by
`tests/execution/test_execution_imports.py` into `bot_engine`. The directly
ordered execution-imports and correlation tests pass (10 tests), and the
leaking test now asserts that `bot_engine.np` remains canonical NumPy.
Full CI run 35771599946 passed: 7066 tests passed, four skipped, zero failures,
80.04% coverage. CodeQL, Workflow Lint, and SBOM also passed on that SHA.

After 20:00 UTC, broker clock confirmed closed, paper broker reported zero
positions/open orders, and the release/backups/research hashes passed preflight.
Production main fast-forwarded from bde5a6a4f to 43abd3d0f; local notes were
preserved in stash@{0} and /tmp/sep21-production-predeploy/local-work.tgz.
Service restarted at 20:00:40 UTC and is active with NRestarts=0. After warmup,
health returns 503 with only `required_model_stale` in readiness failures;
`replay_live_parity_gate_failed` remains an attention flag. Import preflight and
paper ExecutionEngine initialization appeared in the journal, with no structured
error-level entries in the checked postrestart window. Non-sending incident
snapshot passed structurally and reported degraded health. No model promotion,
research trial, holdout evaluation, or trading gate change occurred. Next check:
postdeployment paper session behavior during the next regular market session;
do not infer model readiness or strategy performance from abstention.

## September 22 exact NumPy contamination source

Full CI run 35767890187 on 4c8c3ea54 again failed only the bot-engine
correlation test (7065 passed, one failed, four skipped; 80.04% coverage).
JUnit identified the `np.asarray` replacement as
the lambda created inside
`test_execution_engine_real_when_dotenv_unresolved`.
That test installed a fake NumPy module, reloaded `bot_engine` under it, then
reloaded `bot_engine` again before pytest removed the fake. Reproduced with the
ordered execution-imports and correlation tests: one passed, one failed.
Removed the unrelated fake NumPy setup from that test. The complete
execution-imports file followed by the correlation tests now passes (10 tests);
changed-file lint, mypy, compile and five mapped tests pass. Publish and verify
full CI before deployment. Production and trading/model gates remain unchanged.

## September 22 NumPy isolation repair

CI run 35744924423 on 99995f3b4: 7064 passed, one failed, four skipped,
80.04% coverage. The sole failure was the bot-engine correlation test because
`np.asarray` returned a Python list. Two test modules still installed
list-returning NumPy substitutes during import. Shared test setup also took its
module snapshot before preloading real pandas/NumPy, leaving the canonical
NumPy module outside the `patch.dict(sys.modules, clear=True)` restoration set.
Removed both NumPy substitutes, moved the snapshot after dependency preloads,
and added an isolation regression beside the correlation test. Fourteen focused
tests passed; changed-file lint, mypy, compile, and 11 mapped tests passed;
five selected tests passed with xdist after full collection. No runtime or
trading code changed. Publish this test repair and require passing full CI
before deployment. Production service and model/trading gates remain unchanged.

## September 22 final CI isolation follow-through

Published revision 7f18268dd reached 80.06% coverage (unchanged 80% requirement),
7063 passed, two failed, four skipped in CI35648465789. Remaining failures came
from global NumPy substitutes and a sleep-based overlap test reaching broker IO.
Removed dependency substitutes from sentiment/SPY tests; overlap now synchronizes
with events and exercises the real lock against isolated preparation/execution.
Changed-file validator: lint, types, compile and five targeted tests passed.
Nine ordered isolation/correlation regressions also passed before final cleanup.
Next: publish this test repair to main and verify the full CI result. Production
remains at bde5a6a4f: no deployment or restart during September 22 market hours.
After-close deployment remains authorized and pending a passing release. Preserve
local edits using /tmp/sep21-production-predeploy backups; refresh the release
hash and broker/market checks before using its deployment script. Model readiness,
research budgets and holdout boundaries remain unchanged.

## September 21 CI, deployment, and operating-status follow-through

User authorized items 1–3: repair CI/80% coverage, deploy validated main after
market close, and distinguish paper diagnostics from model readiness. Work is
in /tmp/ai-trading-workflow-fix; production has not yet been restarted/deployed.
Baseline CI35644275203 measured78.97%,6885passed/8failed/4skipped. Fixed test
preflight-state isolation and offline startup equity; removed import-time Flask
and dependency stubs. Cycle budget uses one deterministic clock and no broker IO.
New regressions also found/fixed corrupt-pickle recovery and flat-table bulk
daily grouping. Added meaningful order/fill, persistence, provenance, scheduling,
source-audit and research-budget tests. Gates and coverage threshold unchanged.
Shared health now explains paper_diagnostics_only; close recap repeats it while
required_model_stale remains degraded. No new model trial or holdout evaluation.
Validation: changed-file lint/types/compile,232tests,live health and non-sending
incident check passed; additional research/strategy/cache tests and types passed.
Full CI on the publishing revision remains pending. At20:00 broker clock confirmed
market closed,account active,zero positions/orders; model remained stale.
Next: publish to main, verify full CI coverage, preserve local production edits,
fast-forward deployment after close, restart and verify health/logs. Historical
ledger discrepancies and absent qualified model remain explicit limitations.

## September 21 replacement-model trial preflight

Superseded by completed trial: docs/MODEL_REPLACEMENT_20260921.md. One trial used,
status hypothesis_rejected. 113,223 derived rows; 94,200 OOF opportunities; 30,802
selected proxies; -3.197464 bps/common opportunity, -9.778623 bps/selected trade
after 10 bps costs, 0/5 profitable folds. Do not rerun or tune the consumed trial.
Artifacts: production checkout artifacts/model_replacement_20260921/. Ledger,
report and prediction hashes verified. Cache signatures now exclude audit time;
original manifests preserved and identical builder code/source hashes verified
before recertification. Eleven focused tests, lint/types pass; standard validator
663 tests passed before the cache correction. Direct health and non-sending
incident checks pass structurally; required_model_stale remains. No model saved,
registry activation, restart or holdout evaluation. Runner is on origin/main and
in /tmp/ai-trading-workflow-fix; production checkout was not deployed/reset.

Original preflight history follows:

User explicitly approved one bounded replacement trial; specification is frozen
in config/model_replacement_campaign.json and registered with zero trials claimed
in artifacts/model_replacement_20260921/campaign_state.json in the production
checkout. Scheduled training remains paused. See docs/MODEL_REPLACEMENT_20260921.md.
Nine governed 2024–2025 stock feature checks pass same-history, prefix causality,
and default ten-day runtime-window equality. The older 200-bar truncation failures
are not evidence that the ten-day runtime window fails. Three focused tests pass;
lint/type/compile pass. No model fitted, holdout evaluated or service restarted.
Next: implement provenance-preserving derived features/labels and frozen purged
evaluation before claiming the trial; existing broad trainers are not substitutes.


## Efficient ledger and coverage follow-through — September19

Full CI35411547532 on4ae076049 completed:78.99%,6880pass/4fail/4skip. Nine prior
sklearn failures gone. Four new failures isolated with test-only fixes: throttle
filter, network-free minute fallback, sizing cache, shared order tracking.
Follow-up22parallel tests pass; validator lint/types4/compile/nine tests pass.
Final publishing-triggered CI pending; latest measured coverage remains78.99%,
below unchanged80%. Conserve user budget; do not start another coverage campaign
without a new task. Service and original runtime ledger remain unchanged.

See LEDGER_AND_COVERAGE_20260919.md. Removed global import-time stubs from three
test modules, added fail-closed optional broker-ledger rebuild to the accounting
CLI, and targeted schema/ledger regressions.66 parallel isolation tests pass;
validator lint/types6/compile/40tests pass;17 final ledger/CLI tests pass.
Rebuilt50 broker executions from Sep7 verified opening through Sep18 closing:
quantity matched. Separate ignored artifacts in production checkout under
artifacts/ledger_rebuild/20260918. Original history unchanged, earlier history
unverified/excluded from this rebuild; no live/training source switch or restart.
One full CI measurement follows publishing; do not claim80% from focused tests.
User requested efficient use of remaining26%weekly allowance. Keep future runs
bounded, reuse unchanged validation, and do not delete archived branches.

## September 19 reconciliation and release follow-through

See RECONCILIATION_REPAIR_20260919.md. Current retained ledger reproduces Sep18
two blocks: AAPL-3/AMZN2/MSFT1 versus session-implied0/1/1. Removed unsafe
broker-to-itself reconciliation fallback; fixed separate lineage quantity
overwrite and duplicate/partial audit-to-meta FIFO conversion. Historical data
unchanged; exact incident input snapshots unavailable. Do not claim ledger
repaired. Removed sklearn estimator stubs causing eight parallel CI failures.
104 reporting/conversion tests and56 CI-isolation tests pass. Latest full-deps
suite6827pass/4skip but78.98%coverage;80%threshold remains enforced.
Merged committed close/health fixes bde5a6a4f into isolated worktree. Production
checkout/service untouched; active,NRestarts0,flat,fresh broker,model stale.
Current validation artifacts:/tmp/sep19-*.log. Deployment requires remaining
CI/coverage and evidence review; no restart authorized by passing targeted tests.
Final standard validator --skip-runtime-smoke passed lint,types8,compile and108
tests. Initial imported dataclass-slots diagnostics did not recur on final run;
clean production baseline types2 and scoped changed-file types8 also passed.
Two-worker CI reproduction56pass; integrated replay/health/recap107pass; final
converter group45pass. Counts overlap. Nonsending incident/live health passed.
Saved15:26 report replay now reports2 mismatches instead of the old false zero.
Full-suite CI on the new commit remains required; coverage is not claimed fixed.

## Main integration — September 18

User explicitly requested all current commits on main after disclosure of the
coverage gate. Integrated origin/main into the repair history, resolving only
workflow artifact paths and the repair report; actionlint/diff checks passed.
Normal fast-forward push published 02d5ead2362138afce8a941d80dcda79fa3bac4d to
origin/main. GitHub marked PR3164 merged. No service restart/deployment or local
production checkout reset; existing uncommitted close/health changes preserved.
GitHub accepted the push with required checks pending; CI35301288534 is running
on the exact main commit. Workflow Lint/SBOM passed; full CI not yet verified.

Branch audit:3148 remote refs,104 historical tips outside main ancestry after
integration:7 exact tips linked to merged PRs,87 closed-unmerged PR tips,10
unmatched tips (nine with older PR history plus gh-pages). User explicitly chose
to leave historical branches archived. No historical merge or deletion performed.
All commits from the current repair branch are reachable from origin/main.
Audit artifacts:/tmp/main-branch-audit.json and /tmp/main-closed-prs.jsonl.
Coverage78.82% is the prior complete run's measured statement execution, not a
test pass rate; 6785 tests passed but the unchanged80% minimum failed. Earlier
push-blocker notes below are superseded by this authorized successful push.

## September 18 fixes follow-through

Isolated repair branch/worktree: codex/fix-actions-20260917 at
/tmp/ai-trading-workflow-fix, draft PR #3164 (unmerged, undeployed).
cf7a973e8 fixes discarded immutable TradingConfig.update returns in four test
modules and hidden `.ci/*.xml` artifact uploads. f9fe9ba2d adds fail-closed startup
reconciliation and rejects invalid sliced-order types before submission, with
boundary regressions. Changed-file validator passed lint/types/compile and 43
tests; final startup retry suite nine passed; offline replay/execution 63 passed.
Counts overlap. CI run35295591496 remains in progress at this checkpoint.
Push of f9fe9ba2d was rejected by automatic approval review: remote code export
authorization was not established. This commit remains local; remote PR head is
cf7a973e8. Ask explicit approval to push f9fe9ba2d to the existing origin branch.
Last complete full-deps suite: 6785 passed, coverage78.82% (80% gate unchanged).
Coverage remains unfinished; do not claim all workflows green.

Preflight job33a17762-62de-40ad-a7e7-18f1c8b2112a now has a direct command argv
payload running the canonical module with repo venv and cwd; schedule/delivery/
enabled state preserved. Direct non-sending run complete/blocked, exit0. Next
natural cron delivery remains pending. No service restart; active/NRestarts0.

Fresh two-day broker fetch at01:35:38Z:16 fills, zero execution-linked fee amounts;
one historical cost-comparison pair, zero net-cost pairs. No evidence fabricated,
no qualification gates changed, no training or new experiment. Artifacts under
/tmp/next-fixes-*. Legacy audit-to-meta conversion also produced duplicate
opposite-reward rows for a round trip during testing; not repaired, investigate
before training resumes. Existing close-recap/health edits below remain separate.

## Close recap reporting repair — September 18, 00:05 UTC

Fixed scripts/openclaw_market_close_recap.py: current reset automation/paper
review replaces intentionally paused legacy trading-day/control-plane artifacts;
missing/stale reset evidence still reports pending. Recognize incident checked_at,
use New York session dates (including UTC midnight), constrain last fill to that
session, filter full-session journal before bounded tail, and label edge sums as
unweighted diagnostics rather than net P&L. Healthy close requires connected
broker, flat exposure, no readiness failures and active service.
Canonical health_payload.py now prioritizes degraded readiness and its reason
over the market-closed healthy shortcut. This runtime patch is NOT loaded into
the existing process; it takes effect on the next planned restart. User requested
leave service running; no restart, training, model swap or gate relaxation.
Recap script is immediately used by enabled OpenClaw command job515d67cc-
ed31-41cb-acc8-5e02405d907a (16:45 America/New_York); verified live argv/cwd.
Required changed-file validator passes93tests, lint/types4files/compile. Final
recap-only changes checked separately:13tests, lint/types. Live read-only preview
/tmp/close-repair-preview-final.txt finds8fills and current Sep17 evidence_pending
report. Non-sending incident snapshot passed. Service active/NRestarts0; broker
connected, positions0/orders0. Model age62days exceeds14, abstains; research
reset deliberately pauses scheduled training. Fees missing for8fills; net-cost
and execution-comparison evidence remain unqualified. Do not suppress these.
Logs /tmp/close-repair-validation.log, /tmp/close-repair-recap-final-tests.log.
Next natural close verifies delivery; do not manually trigger notifications.
Rollback the four changed code/test files together; no runtime artifact rewrite.

## Workflow repair PR published — September 17

User explicitly authorized commit/push/draft PR after earlier auto-review block.
Repair branch codex/fix-actions-20260917 pushed; current head eaa952a8f.
Draft PR https://github.com/dmorazzini23/ai-trading-bot/pull/3164 is open.
Worktree /tmp/ai-trading-workflow-fix is clean; temporary venv symlink removed.
First complete PR CI35237366997: 6773 passed, nine failures, coverage78.76%.
Manual full-deps CI35237357129: 6777 passed, five failures, coverage78.80%.
Fixed feature cadence/OHLC fixtures, nested-call regex guard via AST, health
response adapters and stale class fixtures; defer SDK imports in retry and both
reconciliation modules. Added lazy import regressions for four modules.
Follow-up validation: 89 focused tests and seven xdist tests pass; required
agent_validate_changed --market-hours --skip-runtime-smoke passes 436 tests,
lint, types11files and compile. Logs /tmp/workflow-repair-validation-final.log,
/tmp/workflow-repair-focused-final.log and /tmp/workflow-isolation-xdist.log.
Live health responds degraded required_model_stale; non-sending incident passes.
No merge/deploy/restart or installed production dependency changes.
Current head PR CI35242203178 and dispatched full-deps CI35242253502 pending.
Other checks passed on previous head, audit140packages zero findings; verify
new runs before claiming green. CI now uploads coverage XML with JUnit.
Coverage remains unresolved against80%; do not lower threshold/exclude code.
Next: inspect current CI results, download JUnit/coverage, fix remaining failures
and add meaningful coverage for the exact gap. Old logs: /tmp/workflow-pr3164-
{ci,nightly}-failed.log. PR body updated via REST (gh pr edit hits deprecated
Projects classic GraphQL). /tmp/workflow-pr-body.md has current review details.

## GitHub workflow repairs — September 17

See GITHUB_WORKFLOW_REPAIR_20260917.md. CI failures trace to async limit fixture;
fixed observed-time/price assertions,64 targeted tests pass. Dependency Audit
fixed via patched runtime/ML pins and regenerated lock:125packages zero findings,
no ignores. Full dev/test dependency dry-run passes; actionlint passes. Added
constraint syntax/version regression. CI now collects all failures/JUnit while
preserving80%coverage. No production installed dependencies changed or restart.
Publishing was subsequently explicitly authorized and completed; see current
state above. Do not ask again for repair-branch publishing authorization.

## Market preflight parsing repair — September 17, 14:19 UTC

ai_trading/tools/market_preflight.py reads canonical health with urllib and emits
bounded JSON. Valid blocked HTTP200/503 reports exit0; malformed/transport/other
HTTP failures emit failed/unknown and exit1. No jq or guessed nested field paths.
Regression10pass; changed-file validator --market-hours --skip-runtime-smoke:
233passed, lint/types9/compile passed. Live command complete/blocked HTTP503 for
existing required_model_stale + replay parity flags; non-sending incident passed.
Updated OpenClaw job33a17762-62de-40ad-a7e7-18f1c8b2112a via matching
/home/aiuser/.local/bin/openclaw (2026.7.1). Readback matches prompt (gateway trims
trailing newline); schedule/delivery/enabled/identity preserved. PATH's older
.npm-global CLI2026.6.11 remains unchanged. No manual cron run/notification sent.
Next natural scheduled run verifies delivery. No restart/gate/training changes.
See docs/MARKET_PREFLIGHT.md. Logs /tmp/market-preflight-{tests,validation}.log;
live.json, incident.log, job-before.json and job-after.json under same prefix.
Rollback job message from job-before.json via matching CLI, then revert only the
new preflight module/tests/docs. Prior working-tree changes are unrelated.

## Replay tests and pending evidence — September 17, 03:55 UTC

See REPLAY_TEST_AND_PENDING_EVIDENCE_20260917.md. Four replay failures resolved
through test fixture/assertion corrections: later actual observations for limits,
plateaus for markouts/opening policy, candidates rather than fictitious fills
for rising-price duplicate-timestamp case. Full test_offline_replay.py:30 passed.
Changed-file validator:223 passed, lint/types7/compile passed; diff check passed.
No new runtime behavior changes. Healthy active NRestarts0; no new natural skew
or closeout event in postdeployment journal. Both observations remain pending.
Original manifest/report still lack full input contract. Canonical comparison
returns unverified, no qualification authority. No artifact/gate/feed changes.
Logs /tmp/three-{replay-tests,followup-validation}.log; runtime/provenance evidence
in report. No restart/training/new trial/background monitor. Prior four-failure
notes below are historical and superseded by this follow-up.

## Shared five-minute history validation — September 17

Implemented validate_day_sleeve_history in features/day_sleeve.py, extracted
unchanged live timestamp rules. After-hours datasets and replay-aligned features
now reject intraday gaps before indicators. Replay caches versioned v3 and cache
hits validated. Offline replay applies this rule to models declaring 5Min via
model attribute or artifact metadata; other timeframes remain supported. New
replay-trained models declare the validated timeframe; old models not relabeled.
No training, feed/universe/gate changes or restart. Current live rule unchanged.
Regression covers gap/duplicate/order/grid/timezone, session boundaries, cache
bypass, declared replay models and new model timeframe metadata.
Changed-file validator:193 passed, lint/types6/compile passed. Final focused12
passed; final lint passed. Extended suite190pass/4fail; same4 reproduced using
HEAD offline replay function: netting reductions, markout metrics, opening-only
quantile, duplicate-timestamp model scoring. Existing failures remain unresolved.
Logs /tmp/iex-history-{validation,final-tests,baseline,final-regressions}.log.
Live health healthy active NRestarts0, existing stale/parity flags. Natural skew
and closeout-with-exposure still pending. Partial-session boundaries and missing
whole sessions are not certified by continuity alone; full provenance remains
unverified. Rollback: revert these code/test hunks, preserving previous docs;
no data migration. Old cache version is excluded, not deleted.

## IEX-only follow-up — September 17, 03:45 UTC

User declines paid SIP; retain IEX. See IEX_PROVENANCE_AND_GAP_REVIEW_20260917.md.
Original July17 training report/manifest agree on IEX and dataset fingerprint;
full adjustment/session/history/finality contract remains absent. No relabeling.
Unresolved mismatch: serving rejects missing intraday five-minute intervals;
training permits aligned gaps, excludes crossing labels but computes indicators
over surviving rows; replay feature helper lacks serving continuity validation.
Next correction: shared history eligibility with synthetic regression, preserve
strictness; no training/new trials. Audit only, correction not implemented.
23 contract/serving +1 missing-bar training tests passed. Active healthy service,
NRestarts0; stale/parity flags remain. No natural new skew/closeout in postdeploy
journal. No feed/ticker/gate changes, restart or background monitor.

## SIP entitlement/input contract — September 17, 03:35 UTC

See docs/SIP_ENTITLEMENT_REVIEW_20260917.md. SIP latest quotes and recent minute
bars both rejected: subscription does not permit recent SIP. Historical SIP
success is insufficient. /tmp/sip-access-review.json. No subscription/feed edits.
Selected day model 236ba0fe registry declares IEX/5Min/feature version/lookback60,
but lacks full input contract (adjustment/session/history semantics/finality).
Referenced artifact manifest also lacks contract; parity remains unverified.
Do not infer old training settings from current env or substitute generic model.
Natural skew capture and closeout with exposure remain pending; no forced trades.
Read-only checks/docs only; no restart, training or gate changes.

## Skew evidence deployment — September 17, 03:32 UTC

See docs/SKEW_EVIDENCE_AND_FEED_COMPARISON_20260917.md. Skew breaches now record
feature values/reference stats, frame datetime label versus capture time, model
class and actual in-memory joblib SHA1 fingerprint (NOT artifact-file checksum).
Fingerprint failure explicit; predictions/thresholds unchanged. Regression14 pass.
Initial validator343pass/6 stale lifecycle mocks failed; changed those to real
normalizer. Final validator20 pass, lint/types3/compile, non-sending snapshot pass.
Same-window IEX/SIP: AMZN Sep15 389/390 vs390/390; MSFT Sep16 387/390 vs390/390.
No feed/ticker/model/training changes. Historical SIP != real-time entitlement.
Restart03:31:15 succeeded. At03:32:40 healthy/ready cycle1 broker fresh connected,
zero positions/orders; startup logs no warning/error, existing parity/stale flags
remain. /tmp/skew-fix-{posthealth.json,startup.jsonl,validation-final.log};
feed result /tmp/skew-fix-feed-comparison.json. Natural skew warning capture pending.
Docs-only/diff checks passed. Rollback only diagnostic hunks; no data migration.

## Minute gaps/skew — September 17, 03:23 UTC

See docs/GAP_AND_SKEW_REVIEW_20260917.md. 187gap warnings repeat four timestamps:
AMZN Sep15 16:42 and MSFT Sep16 16:29/17:13/18:17. Direct same-feed IEX/all requests
reproduce all four absences; no solely local cache defect established. QQQ skew
16:50:46: RSI/ATR/SMA200/ATRpct/centeredRSI, 5/12=41.7% >35%; meanz1.206<2.5.
OR threshold explains warning; exact historic feature values/model identity absent,
so cause beyond distribution excursion unverified. No runtime/gate/feed changes.
Eight Sep16 intents strategy=day qty1; journals preserve requests2/6/10 versus
submitted1, pending filled0 correctly precedes durable FILLED. Natural reporting
verification now observed. Artifacts /tmp/gaps-skew-{provider,amzn-prior}.json.
Docs-only/diff validation; no restart or new tests needed for read-only review.

## Replay/fee/session trace — September 16, 04:48 UTC

See docs/REPLAY_FEE_SESSION_REVIEW_20260916.md. Exact normalized input and full
reproduced output hashes match Sep15 saved replay. All24 missing comparisons:
10Sep15 fills after cutoff, 9cap-zero, 3simulated orders expired unfilled, 2older
market exits absent decision/TCA input. Candidate90fills/90markouts/no exclusions;
baseline536fills/527markouts (8horizon,1no-subsequent). No source/pricing repair.
Fee-specific FEE/PTC/PTR reads empty+pagination complete. Official current Alpaca
paper docs exclude regulatory fees; do not expect paper data alone to close real
total-cost gate. Account activities may post next day; empty != zero fees.
Updated Sep15 session CLI: all10 linked/quantities/positions matched, fees_missing10.
Artifacts /tmp/replay-missing-observations.json, replay-trace-reproduction.log,
replay-fee-followup-broker.json, replay-fee-session-review/latest.json.
Read-only investigation/docs; no runtime changes/restart/training/new trials.
Natural receipt/closeout-with-exposure verification still pending.

## Four evidence priorities — September 16, 04:36 UTC

See docs/EVIDENCE_FOLLOWUP_20260916.md and TRAINING_RESUMPTION_REQUIREMENTS.md.
Sep15 ten fills: all decision/order/TCA links and quantities/position boundaries
match; sole session gap is ten missing fees. Fresh 2-day broker activity capture:
14fills/14quantitymatches, no fee fields or fee activities. No historical repair.
Fixed reconciliation accepting fees without USD/per_fill_total; stopped comparison
publishing fee bps for invalid fee contracts. Added separate evidence gap counts,
bounded per-order comparison exclusions and explicit all-history scope. Two
benchmark mismatches are Sep11 AMZN255.20vs255.17 and Sep14 AAPL333.02vs332.94;
Sep15 has no matching markout observations. Prices remain unchanged/gates intact.
Validated455 selected +34focused, lint/types39/compile, non-sending snapshot pass.
CLI refreshed /tmp/evidence-four-review.json: fees_missing10, zero comparable pairs.
04:34 broker: no orders since reporting deploy, zero positions/openorders, closed.
Natural production field verification PENDING next eligible order (open13:30UTC).
No forced trades/training/holdout work. CLI-only changes need no service restart.
Artifacts/logs /tmp/evidence-four-*. Training checklist preserves explicit review.

## Reporting provenance — September 15, 18:40 UTC

See docs/REPORTING_PROVENANCE_20260915.md. Primary netted sleeve strategy ID now
passes to durable intents. Receipts separate requested/submitted/filled quantities;
unknown broker quantity stays null; intent quantity remains requested. Historical
records unchanged, gates/research untouched. Focused54 and final contract32 pass;
validator426 pass, lint/types35/compile pass, final types2 pass, non-sending snapshot
pass. /tmp/provenance-{validation,focused,contract-tests}.log. Broker18:38: ten
orders, zero positions; 10/10 durable fill quantity matches, old strategy IDs null.
Restart18:40:05 UTC succeeded; health18:40:39 broker fresh/connected zero positions
and orders, NRestarts0. Startup warnings: existing parity, stale model, minute gaps,
sampling block; no new reporting error observed. /tmp/provenance-post-health.json
and /tmp/provenance-postrestart.log. Docs-only/diff checks passed.
EOD observation pending19:55–20:00; reconciliation command in report after20:15.
No background watcher installed. Natural orders required for production field proof.

## Trade/exit review — September 15, 2026

See docs/TRADE_AND_EXIT_REVIEW_20260915.md. Completed ten-trade audit and explicit
per-trade authority report (/tmp/four-improvements-trade-review.json). Four round
trips -5.31 gross; all tagged stale_model_paper_diagnostic, not qualified ML.
Overnight policy enabled (5-minute lead) DID trigger Sep14; Alpaca 504 escaped EOD
exit and crashed service at20:00 before second symbol. execution_flow now handles
APIError/transport failure per exit, continues, uses stable date/symbol/side client
IDs across cycles/restarts; position-fetch APIError handled. Broker latency can
still defeat close deadline; no guarantee of flatness without broker confirmation.
Three unmocked-network tests repaired: actual imported getter patched; memo test
primary call mocked and historical window aligned. Focused73 passed; validator369
passed, lint/types28/compile passed; non-sending incident smoke passed.
Additional gaps documented: null intent strategy_id; AMZN receipts show requested
8/12 vs actual1; review separates these. 113bps carried-MSFT exit quote needs causal
review. No trading thresholds, research budgets or holdout changed. Restart17:00 UTC;
at17:01 service active/ready, broker fresh, zero positions/orders; only existing
parity/stale-model flags. /tmp/four-improvements-health.json. Docs/diff checks pass.
Logs: /tmp/four-improvements-{focused,validation}.log. Actual EOD window pending.

## Log diagnostics — September 14, 18:40 UTC

See docs/LOG_DIAGNOSTICS_20260914.md. Renamed stale-data rejection warning;
gap logs now identify missing timestamps/provider/feed; skew logs identify outliers.
49 focused tests passed; lint/mypy22/compile passed. Selected suite 648 passed,
three known unmocked daily-fetch DNS failures (get_daily_df unchanged from HEAD).
Non-sending incident check and docs/diff checks passed. Restarted 18:36:30 UTC;
watched through 18:40:26, three active cycles, broker fresh, two positions/zero
orders. No new operational error; only four gap/four replay warnings and existing
stale-model startup error/unavailable warning. No quote/skew recurrence in window.
Same-feed requery confirms AMZN bars absent at 16:35,17:56,17:59 UTC; provider gap
unresolved, not fabricated or hidden. Skew/freshness causes not proven resolved.
Private logs /tmp/log-fix-{validation,tests,postrestart}.log; health
/tmp/log-fix-health.json. Thresholds/models/trading authority unchanged.

## Historical integrity / active reconciler — September 14, 14:29 UTC

See docs/HISTORICAL_INTEGRITY_REVIEW_20260914.md. Read-only audit: 2,082 intents,
715 fill rows; 711 quantities match complete May 3 onward broker capture; four
unmatched explicitly tagged cutover drills. No observed duplicates/overfills,
orphans, invalid quantities, post-terminal live-source events or reopened intents.
SQLite quick_check ok. Repair preview: no database mutations justified.
LiveTradingEngine durable/pending reconciliation confirmed in logs, zero lookup
errors. Optional PositionReconciler worker not wired into repository startup;
no activation observed. Session now open; final broker fresh, zero positions/orders;
health degraded with existing parity/stale-model flags. Post-close audit pending.
Private evidence /tmp/history-integrity-audit.json and /tmp/history-audit-*.json.
No runtime edits, trades or restarts. Docs-only validation/diff check passed.

## Recovery fixes — September 14, 2026

All four audited bugs corrected; see docs/ORDER_RECOVERY_FIXES_20260914.md.
Conditional terminal transitions; cumulative fills serialized in DB across store
connections/restarts; invalid snapshots preserve state; APIError recovery and
interruptible worker stop/restart. New regression suite and manager fake updates.
Focused: 38 passed. Validator: 247 passed, lint/types (17 sources)/compile passed.
Synthetic lifecycle parity: four scenarios, zero mismatches. Non-sending incident
check passed. Service restarted; at 03:23 UTC active/ready, broker fresh/connected,
zero positions/orders; only existing stale-model/parity flags. Snapshot:
/tmp/recovery-fix-postdeploy.json. Docs-only validation and git diff --check passed.
Logs: /tmp/recovery-fix-{tests,validation,parity}.log. No trades/gate changes.

## Recovery audit — September 14, 2026

Historical findings, now fixed above: docs/ORDER_RECOVERY_AUDIT_20260914.md.
P1 terminal intents reopened by late callbacks; P1 concurrent cumulative fill
callbacks double-count; P2 None snapshot clears optional reconciler state;
P2 APIError kills optional reconciliation loop with running flag still true.
Five local reproductions passed; retained fixture in
artifacts/audits/test_order_recovery_20260914.py asserts BUG behavior, not fixes.
Existing focused tests: 17 passed. No runtime changes, trades or deployment.
Live manager wiring verified for OMS findings; optional reconciler activation
not established. Next implementation priority: atomic durable transitions and
fill deduplication, then snapshot failure and worker recovery handling.

## Operational review — September 14, 2026, 02:58 UTC

See docs/OPERATIONAL_REVIEW_20260914.md. Fresh health: broker connected/fresh,
zero positions/orders; existing stale-model/parity flags. Market closed;
post-deployment regular-session verification remains pending (13:30–20:00 UTC).
Verified daily Sep 11, weekly Sep 12 and Sunday Sep 13 workflows/operator reports;
three reset-approved evidence steps passed. Training guard research_reset_active.
Policy/scorecard tests: 22 passed. Fresh accounting: 535 activities, 407 quantity
matches, 407 unknown fees; Documents API access not available in configured client.
Fresh scorecard: artifacts/research_reset/operational_review_20260914_scorecard.json;
12 observed fills, zero complete chains, missing causal quotes/fees for all 12.
No runtime changes or trials. Sep 15–21 review pending future weekly Sep 19 output.
Existing one-shot user verification timer expired Sep 8; recurring evidence timers
remain active. No automatic follow-up added. Next: inspect regular-session evidence,
then next weekly report; obtain authoritative per-fill fee records if available.

## Deep input fixes — September 13, 2026

See docs/DEEP_INPUT_VALIDATION_FIXES.md. Strict input-contract scalar/nested schema
validation; malformed contracts cannot match. Live feature builder rejects invalid
historical OHLCV and duplicate/naive/off-grid/irregular timestamps before indicators.
Replay training retains missing features instead of ffill/zero; RSI errors propagate
in both replay and after-hours paths. Feature-cache v2 prevents old imputed reuse.
Shared training/label_timing.py rejects gaps, cross-session/holiday/early-close and
off-grid labels. Shadow overrides enforce elapsed horizon/session too. Warmup and
session-boundary exclusions are explicit; numeric quality thresholds unchanged.
Empty datasets retain quality diagnostics. Existing models and research gates intact.
Execution evidence refreshed: 535 activities, 407 quantity matches, 407 unknown
fee totals. /tmp/deep-input-accounting.json; no fee fabrication or trades.
Real development feature check: 9/9 same-history, 9/9 causal. Ledger hashes unchanged.
Validator: 199 selected tests, lint/types (10 sources), compile passed;
/tmp/deep-input-final-validation.log. Additional final focused tests: 46 passed;
after-hours/helper scope: 114 passed. Final type check: 2 sources passed;
/tmp/deep-input-last-types.log. Deployed by service restart September 14.
At 02:31 UTC service active/ready; broker fresh/connected, zero positions/orders.
Only existing replay_live_parity_gate_failed and required_model_stale flags remain.
Health snapshot: /tmp/deep-input-postdeploy.json. Non-sending incident check passed.
Regular-session inference and scheduled-training behavior remain to be observed;
no forced trades/training or research-budget changes. Per-fill fees remain blocked.

## Calendar and input-contract implementation — September 13, 2026

See docs/INPUT_CONTRACT_IMPLEMENTATION.md. Intraday normalization now uses the
canonical exchange calendar; holiday/early-close rows excluded. Finalization intact.
Provider/request attributes, finalized-batch bounds/count/hash and inference
feature hash/column order are logged; decision debug includes input provenance.
day_input_v1 compares explicit model metadata with serving inputs, reporting
unknown/mismatched fields as unverified with no qualification authority. Loader
preserves absent historical contracts as absent. No feed/adjustment/history change.
Review: live IEX/all/10-day request vs incomplete historical model contract;
current reference and development sources differ. Do not infer historical settings.
Focused tests: 31 initial calendar/serving/contract; final 23 integration tests and
16 loader tests passed. Final lint, compile and additional types (3 files) passed.
Changed-file validator: 1,330 passed, 3 daily-fetch tests failed on unmocked Alpaca
DNS requests in sandbox. Isolated reruns reproduce; get_daily_df AST unchanged
from HEAD. /tmp/input-contract-validation.log; /tmp/input-contract-daily-failures.log;
/tmp/input-contract-fetch-regressions.log. Full suite not clean; no broad rerun.
Non-sending snapshot passed; preflight fresh broker, zero positions/orders.
Paper-service restart succeeded; active, broker fresh/connected at 15:45:59 UTC,
zero positions/orders, no broker failures; only existing model/replay flags.
/tmp/input-contract-health-final.json. Selected-model contract review returns
unverified with missing declarations: /tmp/input-contract-review.json.
Docs-only validator and git diff --check passed after final handoff update.
Natural finalized-batch/eligible-inference provenance verification remains pending
normal scheduled session. Do not force trades or bypass stale-model gates.

## Live/training bar contract verification — September 13, 2026

See docs/LIVE_TRAINING_BAR_CONTRACT_AUDIT.md. Effective runtime settings: 10-day
default live history vs selected model 60-day training metadata; live IEX/all,
current reference-training delayed_sip/raw, development SIP/split. Historical
selected-model feed/adjustment absent; do not infer it from current training code.
Finalization verified at five-minute close + two seconds. 19 serving tests passed.
Synthetic checks confirm normalize_bars retains holiday and post-early-close rows;
it filters weekdays/time-of-day rather than canonical exchange sessions.
No matching bar-fetch evidence in post-reboot service logs; actual returned live
batch remains unverified. Health broker fresh, zero positions/orders, existing
model/replay flags. No runtime edits, settings changes, new fetches, training or trials.
Next corrective scope: calendar normalization, actual inference-batch provenance,
then versioned training/serving input contract. Preserve selected model abstention.
Docs-only validation and git diff --check passed.

## Bounded feature parity audit — September 13, 2026

See docs/STOCK_FEATURE_PARITY.md. New ai_trading.tools.stock_feature_parity checks
current training/runtime builders on governed 2024-2025 stock data only. Nine
fixed prefixes (250/500/1000 bars x three stocks): all identical-history and
prefix-causality comparisons pass. Last-200-bar histories differ in recursive
features at rtol=atol=1e-10; economic/prediction effect not measured.
Report artifacts/stock_development/feature_parity.json. No model loaded, returns,
training, trial consumption, holdout evaluation, orders or gate changes.
Warmup regression confirms training imputation can yield finite rows where runtime
rejects insufficient history. Full live-history equivalence remains unverified.
Focused tests: 2 passed. Changed-file validator passed: 616 tests, lint, types
(7 sources), compile; /tmp/feature-parity-validation.log. Used --market-hours
--skip-runtime-smoke with separate health/incident checks and temporary bytecode cache.
Docs-only validation and git diff --check passed. No runtime deployment needed.
Source/ledger checks, live health and non-sending incident snapshot passed.

## Stock-universe development readiness — September 13, 2026

See docs/STOCK_DEVELOPMENT_READINESS.md. Post-maintenance service active, reboot
00:57 UTC, broker fresh/connected, zero orders/positions; prior model/replay flags.
Acquired governed AAPL/AMZN/MSFT SIP split-adjusted 1Min bars for 2024-2025 only:
artifacts/stock_development/acquisition.json, quality passed, 194,700 expected and
observed regular-session minutes per stock. No holdout acquisition/evaluation.
New CLI ai_trading.tools.stock_development_readiness verifies source provenance
and audits raw OHLCV plus 200-bar/current-session feature history. Canonical source
verification extracted from research_feasibility; existing sampling logic retained.
Report artifacts/stock_development/readiness.json: each stock 37,936 valid windows,
195 warmup exclusions, 37,741 necessary feature-input-supported slots. No missing,
duplicate or invalid regular-session minute exclusions. Full feature parity and
execution remain unverified; feature imputation/RSI fallback cannot prove validity.
Fresh broker accounting: 535 activities, 407 quantity matches, 407 unknown fee
totals, no fee execution linkage. /tmp/stock-audit-accounting.json.
Campaign ledger hashes match prior audit; no trials, fitting, returns or orders.
3 readiness tests passed plus lint. Non-sending incident snapshot passed.
Changed-file validator passed: 613 tests, lint, types (5 sources), compile;
/tmp/stock-readiness-validation.log. Used --market-hours --skip-runtime-smoke,
PYTHONPYCACHEPREFIX=/tmp/stock-audit-validation-cache, separate health/incident checks.
Final added warmup test passed in the 3-test focused run. Docs-only and diff checks
passed. No deployment needed. Next bounded gap is full feature computation parity
and execution provenance; input support alone does not authorize model use.

## Five-minute requirements and coverage — September 13, 2026

Completed requested timing definition and development-only coverage assessment.
See docs/FIVE_MINUTE_TIMING_REQUIREMENTS.md. Proposed convention: completed
five-minute inputs, entry D+1 minute, exit D+6 (five minutes after entry); exact
unique minute coverage throughout, same regular session. Not adopted in runtime,
not retroactive labels, not a new trial. Existing model label durations remain
irregular (fresh metadata check: maximum 89.67 hours).
New reproducible CLI: ai_trading.tools.five_minute_coverage; report at
artifacts/research_reset/five_minute_coverage.json. 2024-2025 governed ETF timestamps
only, verified hashes and quality, no returns/signals/prices used in analysis.
502 sessions/symbol: 38,438 candidate slots, 502 boundary exclusions each.
DIA 36,088 complete, 1,848 missing; other five ETFs 37,936 complete, zero missing.
No duplicate exclusions. Current AAPL/AMZN/MSFT coverage is not available in this
dataset. Counts are timestamp upper bounds, not model-ready or independent trades.
Campaign ledger hashes unchanged; no training, trials, orders or holdout evaluation.
Tests: 2 new regressions passed. Validator lint, types (2 sources), compile passed;
609 tests passed, 2 repository-audit tests initially failed writing pycache into
read-only .codex. Rerun test_audit_repo_tool.py with
PYTHONPYCACHEPREFIX=/tmp/five-minute-audit-pycache: all 3 passed, no source workaround.
Logs: /tmp/five-minute-validation-final.log and
/tmp/five-minute-audit-cache-validation.log. Live broker fresh, existing stale-model/
replay flags only; non-sending incident snapshot passed. No deployment required.
Docs-only validation and diff check passed. Next: separately review applicability
to current stock universe and full feature requirements before any performance study.

## Production timing verification — September 13, 2026

Executed installed replay CLI with runtime environment and production paths;
exit 2, blocked REPLAY_POLICY_NON_REGRESSION_FAILED as expected. New artifact:
/var/lib/ai-trading-bot/runtime/replay_outputs/replay_hash_20260913.json
SHA256 8b133b8ef7d03361a7ecbbd16bfc4621eed24448764bf1803e3b8904fffeb360.
Both summaries persist timing contracts and per-fill diagnostics. Category counts,
diagnostic rows and usable-plus-excluded markouts reconcile to all fills.
Candidate: 82 fills, 76 usable; decision-target missing 34, fill-target missing 44,
78 fills at/after decision target; net edge -10.735105 bps. Baseline: 534 fills,
520 usable; decision-target missing 200, fill-target missing 249.
Rolling source window changed; do not interpret score differences as improvement
or regression caused by diagnostics. No tuning, training, orders or gate changes.
Health 00:41:14 UTC: paper broker connected/fresh, zero orders/positions; only
existing replay/stale-model flags. Non-sending incident snapshot passed.
Logs: /tmp/timing-production-verification.log; /tmp/timing-production-health.json.
Timer active; last actual invocation September 11, next September 14 09:22:18 UTC.
Production CLI integration verified; actual timer-triggered adoption still pending.
Runtime code unchanged, prior 836-test validation reused. Docs-only validation
and git diff --check passed for this handoff update.

## Timing contract implementation — September 12, 2026

See docs/REPLAY_TIMING_CONTRACT.md. Replay summary now persists explicit metric
anchors and exact 300-second observation diagnostics for every fill, including
excluded fills. Diagnostics have no qualification authority or alternate returns.
Existing metrics/gates, costs, model selection and research restrictions unchanged.
Isolated replay /tmp/timing-check/output/replay_hash_20260912.json matched prior
production source hash and every existing summary field. Correctly blocked exit 2.
Candidate exact observations missing: 33/83 decision-anchor, 43/83 fill-anchor.
Baseline missing: 201/539 decision-anchor, 249/539 fill-anchor. Counts reconcile.
32 focused tests passed; final exact-observation assertions: 2 passed plus lint.
Changed-file validator: 836 tests passed, lint/types (9 sources), compile;
/tmp/timing-validation.log. Used --market-hours --skip-runtime-smoke with separate
isolated replay, live health and non-sending checks. Docs-only and diff checks passed.
Health broker fresh, zero orders/positions;
non-sending incident snapshot passed. No service restart needed: scheduled CLI
loads checkout each invocation. Actual timer check remains pending September 14.

## Four-item evidence audit — September 12, 2026

See docs/REPLAY_BLOCKER_SCORECARD_20260912.md for source hashes and completion criteria.
Executed installed replay CLI with production paths/settings; artifact refreshed
17:46 UTC, correctly blocked on qualification. Manual systemd start requires an
unavailable password; actual timer adoption pending September 14 09:22:18 UTC.
Timing: 33/77 candidate markouts exceed five minutes; 47/77 fills delayed beyond
five minutes. Model metadata still has irregular elapsed label horizons.
Fresh broker reconciliation: 535 activities, 407 quantity matches, 407 unknown
fee totals; 83 fee records have no execution references. External evidence gap.
No tuning, training, order submission or gate changes. Documentation only.
12 focused replay-tool/accounting tests passed; artifact arithmetic assertions and
non-sending snapshot passed. Health 17:47 UTC broker fresh, zero orders/positions,
existing stale-model/replay flags. Next task: explicit governed timing contract
and coverage diagnostics; preserve holdout and consumed research budgets.

## Latest IOC implementation — September 12, 2026

See docs/REPLAY_IOC_FIXES.md. Simulator IOC uses a submission-time observation,
cancels any remainder and never waits for a later quote. Replay applies immediate
fills to exposure before the next cap check; summaries persist cancellations.
Qualification/cost/model/research gates unchanged. Earlier IOC blocker below is
resolved: isolated replay now writes an artifact and correctly fails qualification.

- Focused regressions: 51 passed; /tmp/replay-ioc-regressions.log.
- Isolated artifact: /tmp/replay-ioc-check/output/replay_hash_20260912.json.
  Candidate 77 usable + 6 excluded = 83 fills; baseline 525 + 14 = 539.
  All bounded markouts fill before expiry; three baseline IOC fills have zero delay.
  Candidate -9.750137 bps; positive edge, sample minimum and non-regression fail.
- Non-sending incident snapshot passed. Preflight 17:40 UTC: paper broker fresh,
  zero orders/positions; existing stale-model/replay attention flags only.
- Changed-file validator passed: 834 selected tests, lint, types (8 sources),
  compile; /tmp/replay-ioc-validation.log. Used --market-hours --skip-runtime-smoke
  with separate live health, incident snapshot and isolated replay checks.
- Paper service restart succeeded; service active and broker fresh/connected at
  17:43:03 UTC, zero positions/orders and no broker failures. Only existing
  stale-model/replay flags remain; /tmp/replay-ioc-health-final.json.
- Docs-only validator and git diff --check passed. Next: address named evidence
  gaps under research-reset governance; no qualification or promotion justified.

## Latest expiry implementation — September 12, 2026

See docs/REPLAY_EXPIRY_DIAGNOSTICS.md. Day/GTC and explicit expiry are propagated
through replay; governance missing TIF uses labeled day assumption. Day expiry
uses canonical NYSE close, precedes fills and preserves partial filled quantity.
Expired events and per-fill markout exclusion reasons persist in summaries.
No qualification, cost, horizon, model or research-setting changes.

- Changed-file validator: 827 selected tests passed, lint/types (8 sources), compile.
  /tmp/replay-expiry-validation.log.
- Focused regressions: 44 passed. /tmp/replay-expiry-regressions-final.log.
- Final simulator lint/compile and git diff --check passed.
- Initial isolated-run approval hit a usage limit. After its reset and user
  continuation, the normal approval path succeeded; no workaround used.
- Isolated replay then failed on unsupported recorded time_in_force=ioc before
  writing an artifact. /tmp/replay-expiry-check.log and
  /tmp/replay-expiry-check/summary.json. Do not report real-data expiry reconciliation
  as verified. IOC requires evidence-backed handling; no approximation introduced.
- Non-sending incident snapshot passed. Preflight 17:32 UTC: paper, fresh broker,
  zero positions/orders; existing stale-model/replay flags only.
- Service restart succeeded; active service and fresh broker at 17:34:15 UTC,
  zero positions/orders and no broker failures. Existing stale-model/replay flags
  remain. /tmp/replay-expiry-health-final.json.
- Documentation validation and final git diff --check passed.

## Latest evidence review — September 12, 2026

Completed four-item post-fix review: docs/POSTFIX_REPLAY_EVIDENCE_REVIEW.md.
No new scheduled artifact existed; systemd start required a sudo password.
Ran the same CLI/settings as aiuser with isolated /tmp/postfix-replay-check outputs.
Production artifact/service untouched. CLI wrote evidence then correctly blocked
on REPLAY_POLICY_NON_REGRESSION_FAILED.
68/68 candidate markouts passed field/timestamp/assumption/arithmetic checks.
7,396 shadow records rejected for missing/unsupported types; final 1,859 source
rows reduce to 911 by configured symbols. Candidate -10.888324 bps, 68 samples;
positive edge, sample minimum and edge non-regression fail.
Median fill delay 1 hour; median markout interval 10 minutes; maximum fill delay
94.75 hours. Selected model targets one 5Min bar but recorded training label
elapsed times are also irregular. Diagnostic-only, not qualification evidence.
Next gap: order expiry/time-in-force and explicit markout exclusion diagnostics.
No tuning, model fitting, live orders or promotion. Holdout restrictions remain.

## Latest implementation — September 12, 2026

Replay contract fixes implemented; see docs/REPLAY_CONTRACT_FIXES.md.
Recorder/journal and replay separate submission_status from executable type.
Unsupported types fail before simulator order creation; unknown opportunity intent
is rejected diagnostically, never guessed. Explicit historical order_type remains
accepted. Saved markouts include timestamps, intervals and simulator assumptions.
Qualification gates, model selection, costs and research restrictions unchanged.

Changed runtime: contracts/decisioning.py, core/decision_log.py,
core/bot_engine.py, execution/simulated_broker.py, replay/event_loop.py.
Regressions: tests/test_decision_recorder.py,
tests/test_replay_governance_async_parity.py,
tests/unit/test_simulated_broker_reproducible.py.

- Final focused synthetic replay/broker tests: 40 passed.
  /tmp/replay-contract-final-synthetic.log.
- Final recorder suite: 11 passed. /tmp/replay-recorder-final.log.
- Changed-file validation passed: lint/types (8 sources), compile, 823 tests.
  /tmp/replay-contract-validation-final.log.
- Final explicit-type fallback lint/compile passed after its last edit.
- Non-sending incident snapshot passed; git diff --check passed.
- Preflight 02:05 UTC: paper, fresh broker, zero positions/orders. Existing
  provider-backup/degraded and stale-model/replay flags remain.
  /tmp/replay-contract-preflight.json.
- Restart succeeded; active service and fresh broker at 02:10:05 UTC confirmed
  zero positions/orders and no broker failures. Only existing stale-model/replay
  flags remain. /tmp/replay-contract-health-final.json.
- Documentation validation and git diff --check passed.
- No new research replay, historical artifact rewrite, model or order initiated.

## Latest bounded audit — September 11, 2026, 14:58 UTC

Completed execution-price-drag audit: docs/REPLAY_EXECUTION_DRAG_AUDIT.md.
Arithmetic reconciles, but 2512/3174 markouts use not_submitted as an executable
order type (market-like behavior), simulator defaults supply spread/volatility,
and persisted rows lack timestamps needed to validate actual holding intervals.
Synthetic limit-vs-not_submitted reproduction confirmed the semantic difference.
No runtime code/config/model changes or new replay. Next corrective scope is in
the audit; do not infer historical order intent or tune from overlapping holdout.

Updated September 11, 2026, 14:54 UTC. User authorized all three: rollback-counter
repair, existing replay diagnosis, and governed model replacement decision.
Report: docs/ROLLBACK_REPLAY_MODEL_DECISION.md.

## Current implementation

- ai_trading/governance/promotion.py serializes KPI evaluation/accounting/actions
  under a shared file lock; atomic state writes fail closed. Observation identities
  prevent poll/retry inflation across restarts and healthy resets. Unverified
  legacy counts are diagnostic only. Conflicting payload identity is rejected.
- ai_trading/main.py uses persisted counts without increasing a local streak;
  source observation timestamps supply runtime identities.
- ai_trading/monitoring/slo.py exposes last_observation_at.
- Regression coverage: tests/governance/test_kpi_observation_accounting.py,
  tests/main/test_runtime_governance_hooks.py and updated institutional/governance
  fixtures use distinct evidence where a new observation is intended.

## Validation

- bash scripts/agent_validate_changed.sh --market-hours --skip-runtime-smoke:
  lint, mypy (49 sources), compile and 720 selected tests passed.
  /tmp/three-items-validation.log.
- Final targeted rollback/scheduler tests: 14 passed before one additional
  distinct-concurrent-writers test was included in the selected validation.
  /tmp/rollback-targeted-final.log.
- Final lint/compile for updated terminal-status guard and concurrency test passed.
- Non-sending incident snapshot passed. No messages or orders sent by this task.
- Saved replay hash, sample counts, gross-minus-drag identity and attribution
  residual verified directly; no new backtest, experiment or source replay run.

## Evidence decisions

- Latest existing replay (September 11 09:23 UTC): candidate -18.281386 bps,
  3174 next-observation markouts. Gross -3.941979 minus execution-price drag
  14.339406 bps. Baseline also negative (-17.822979); cap contrast -0.458407 bps.
  This is simulated markout accounting, not live P&L or pure fee attribution.
- Source report overlaps protected holdout; do not tune/select from its subgroups.
- Replacement decision: retain abstention. Selected model ~55.8 days old against
  14-day limit and newest of 187 ml_edge registered entries (July 17).
- Latest training report: no_qualified_candidate; logreg post-cost OOF -19.025753
  bps, 0/5 profitable folds, no runtime/promotion/live-money authority.
- Research reset remains active, no gate weakened and no replacement installed.

## Deployment

- Preflight 14:53:38 UTC: paper, broker connected/fresh, one position,
  zero open orders. /tmp/three-items-predeploy.json.
- Service restart succeeded. Active service and fresh broker at 14:54:46 UTC:
  one position, zero open orders, no broker failures. Only existing stale-model
  and replay flags remain. /tmp/three-items-health-final.json.
- Documentation-only validation and git diff --check passed.

## Earlier completed work and constraints

Seven audit fixes: docs/AUDIT_SEVEN_FIXES.md. Shadow-start/provenance follow-up:
docs/GOVERNANCE_FOLLOWUP_FIXES.md. Environment/reset fixes: docs/ENV_RESET_FIXES.md.
All were previously validated/deployed. Preserve these and unrelated dirty
research/efficiency changes. No commit/reset to tidy the tree.
September 9–December 8 holdout and consumed campaign budgets remain. October 8
does not auto-release research. Historical holdout isolation is not established.
Legacy shadow evidence requires review; explicit administrative force promotion
still bypasses ordinary eligibility. KPI source IDs establish deduplication,
not statistical independence; no-ID callers conservatively hash monitored values.
