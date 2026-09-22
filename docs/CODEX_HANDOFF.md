# Current handoff

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
