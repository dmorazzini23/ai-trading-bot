# Replay expiry and markout diagnostics

September 12, 2026. Correctness changes; qualification thresholds remain intact.

Governance replay now explicitly supplies time_in_force. Recorded day/gtc values
and expires_at survive source refresh and normalization. Missing governance intent
uses a labeled governance_day_assumption rather than an implicit perpetual order.
This is a simulation policy, not reconstructed historical intent. The generic
simulator's existing GTC default remains labeled simulator_default_gtc; governance
overrides it with day. Unsupported time-in-force values are rejected.

Day orders expire at the session close from the canonical NYSE calendar, including
early closes and timezone changes. A supplied earlier expiry takes precedence.
Non-session or after-close day orders cannot acquire a later fill. GTC can remain
open unless explicitly bounded by expires_at. Expiry is checked before fills and
applies to accepted, partially filled and never-scheduled orders. Filled quantity
is preserved; expired orders cannot later fill or be canceled as open orders.

Expiry events preserve effective expiry time, observation time, identity, time in
force, assumption source and remaining quantity. Replay summaries persist those
events, including when no markout qualifies. Sparse replay only observes expiry
when its clock advances; observed_at distinguishes that delay from effective expiry.

Markout summaries now persist fill_events_seen, per-reason exclusion counts and
per-fill exclusion records for invalid fill prices/fields, missing subsequent
prices, excessive horizons, nonfinite net markouts and invalid reference prices.
For completed summaries, included samples plus exclusions reconcile to fill count.
Invalid fee contracts retain their existing fail-fast behavior.

Regressions cover early-close/no-late-fill behavior, partial/unfilled expiry,
explicit GTC, unsupported time-in-force, and included/excluded fill accounting.
The first test run caught duplicated expiry events; final code queues each event
once. Changed-file validation passed 827 selected tests, lint, types (8 files),
and compilation. Final simulator lint/compile passed after event metadata additions.

Existing artifacts, research reset, holdout restrictions, costs and qualification
thresholds are unchanged. A new replay can change sample composition because orders
now expire; that is not evidence of strategy improvement. No parameter search or
model fitting is authorized by this correctness change.

Final isolated diagnostic and deployment results are in docs/CODEX_HANDOFF.md.
The isolated runtime-data attempt on September 12 stopped before artifact creation
on recorded time_in_force=ioc. IOC is intentionally unsupported; no substitute
fill policy or filtering was used to manufacture a passing report. Real-data
markout reconciliation remains unverified until IOC semantics have an evidence-backed
implementation or that input is otherwise resolved under the governing protocol.
Rollback only this task's simulator, event-loop, summary and test changes while
preserving earlier contract fixes. Old code permits longer-lived orders; do not
mix its results with new expiry-aware evidence as if execution assumptions match.
