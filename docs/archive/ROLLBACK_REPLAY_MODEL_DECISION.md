# Rollback accounting, replay diagnosis and model decision

September 11, 2026. Scope: correctness repair and review of existing evidence.

## Rollback accounting repair

The persisted counter and scheduler both counted evaluator calls as breaches.
The public evaluation now holds a file lock across accounting and rollback
selection. State uses atomic writes and persistence failures propagate before
rollback. Corrupt state fails closed rather than becoming an empty counter.

Observation identities persist across restarts and healthy observations. Exact
retries do not increment; conflicting payloads under one identity fail. Healthy
observations reset the streak while retaining deduplication history. Legacy counts
are retained as unverified diagnostics and cannot authorize rollback. The scheduler
uses only the persisted count and does not synthesize a larger local streak.

SLO snapshots expose their latest source observation timestamp. Runtime identities
use these timestamps plus drawdown, rather than polling time. Callers without
source identity conservatively deduplicate identical monitored KPI values; distinct
equal-valued observations require a distinct source identity. The count describes
distinct observations, not independent statistical samples or completed sessions.
The persisted ID history is retained and grows with observations.

## Replay diagnosis

Existing artifact: /var/lib/ai-trading-bot/runtime/replay_outputs/replay_hash_20260911.json

SHA-256: d496bd9713ee3530053f5fe7eb9fa706fb42bf822a48ff9071edd236fef1bde2

The report generated at 09:23:36 UTC supersedes the earlier -18.65 bps snapshot.
It reports 3,174 capped candidate markouts and 4,226 uncapped baseline markouts.

| Mean markout component | Candidate | Baseline |
| --- | ---: | ---: |
| Gross reference-to-next-observation markout | -3.941979 bps | -2.476913 bps |
| Execution-price drag | 14.339406 bps | 15.346065 bps |
| Net markout (gross minus drag) | -18.281386 bps | -17.822979 bps |

These means were recomputed from the saved markout rows. Execution-price drag
includes fill/reference-price timing and denominator effects; it is not a claim
that independently measured broker fees equal 14.34 bps. Removing that entire
modeled drag would still leave a negative gross markout.

The saved sequential attribution reconciles the candidate-minus-baseline difference
of -0.458407 bps to the cap-retained order set, with zero residual. Its separately
reported fill and cost contributions to that contrast are zero. Caps fully block
1,110 requests and adjust 1,945. This explains the small policy contrast, not the
whole negative headline. Loosening caps does not establish a positive edge.

Signal selection and simulated execution both warrant scrutiny: the gross proxy
is negative and modeled execution-price drag deepens the loss. Linked exits,
queue position, market impact and separately observed fees are not established by
this artifact. It cannot identify realized strategy P&L or an optimal exit policy.
No new strategy, parameter search, backtest or source-data replay was run.

Source interval in the existing report is August 12–September 10 and overlaps the
protected September 9–December 8 holdout. This review describes already-produced
operational evidence only. Do not use its favorable subgroups to select or tune
a model, and do not claim the historical holdout is uncontaminated.

## Governed replacement decision

Decision: retain abstention; do not replace or promote a model now.

Live health at 14:48 UTC identifies selected model
ml_edge-histgb-236ba0fe-20260717200350-4045d5c5 as stale: about 55.8 days against
a 14-day maximum. It is the newest registered ml_edge entry among 187 entries in
/var/lib/ai-trading-bot/models/registry_index.json, registered July 17. Registration
recency alone is not qualification; no model binaries were loaded in this review.

The latest after-hours report (September 9, 20:00 UTC) says no_qualified_candidate,
with runtime, promotion and live-money authority all false. Its logreg candidate
has -19.025753 bps post-cost OOF expectancy and zero profitable folds out of five.
Neither that candidate nor the stale selected model supplies replacement authority.

The active research reset pauses new models and scheduled training and provides no
promotion authority. October 8 is a review date, not automatic permission to resume.
A later replacement requires explicitly authorized research, governed data with
holdout isolation, positive post-cost out-of-sample evidence, passing confirmation
checks, identified shadow evidence and fresh replay/operational qualification.
Refreshing the model timestamp or lowering gates would not satisfy those requirements.

## Validation and rollback

Regression tests cover repeated and concurrent observations, distinct concurrent
writes, restart persistence, healthy resets, identity conflicts, legacy counts,
failed writes and the scheduler's use of the authoritative count. Final commands
and service verification are recorded in docs/CODEX_HANDOFF.md.

Rollback only this task's changes in promotion.py, main.py, monitoring/slo.py and
their tests, preserving previous fixes. Older code does not understand the new
observation history; do not roll back into automatic rollback authority using
unreviewed state. No environment or model selection changes are required.
