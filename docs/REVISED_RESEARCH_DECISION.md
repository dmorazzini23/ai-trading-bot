# Reversal study decision — September 6, 2026

The revised hypothesis does not qualify. Close both tested variants as
inconclusive because validation support is below the predetermined minimum.
Do not expand this grid or relax the threshold after seeing these results.

The original continuation study's 1-, 5-, and 15-minute variants are now
formally retired in `FOCUSED_RESEARCH_REGISTRY.json`. Its 60-minute variant is
closed as inconclusive. The focused runner checks this registry before loading
data, including when a completed rule is given a different protocol name. It
also refuses to overwrite a completed report. This is a research workflow
restriction, not a change to the live model registry or trading configuration.

## Fixed hypothesis and results

`REVISED_RESEARCH_PROTOCOL.json` was written before the new outcomes. It tests
whether temporary selling pressure produces a tradable reversal. AAPL is
bought after a 15-minute log-price decline of at least two volatility units;
volatility uses the preceding 60 one-minute returns, excluding the shock.
All 76 closes must be contiguous within the same New York session. Entries
start at 11:00, occur at the next minute open, and must exit by 15:00. The only
holding periods are 15 and 60 minutes. There is no fitted parameter search.

The same source manifests yield 285 verified historical sessions, including
170 validation sessions across three periods. These dates were already used
for the continuation study; this follow-up is exploratory, not independent
confirmation. Costs, support requirements, and period stability requirements
were retained. Twelve simultaneous uncertainty bounds conservatively count
both the four original and two revised horizons, each with two metrics; this
does not correct an unknown number of older project searches.

| Hold | Validation trades | Gross bps/trade | Net bps/trade at 12 bps round-trip | Break-even one-way cost | Decision |
|---|---:|---:|---:|---:|---|
| 15 minutes | 37 | 0.942 | -11.058 | 0.471 bps | Inconclusive; closed |
| 60 minutes | 4 | -36.691 | -48.691 | -18.346 bps | Inconclusive; closed |

Both are below the required 250 trades. Neither has a positive validation
period at the primary cost. Adjusted lower bounds on mean daily net returns
are -5.088 and -3.525 bps. A negative break-even cost means the observed gross
return is already negative; it is not an attainable execution assumption.
Beating the high-turnover always-long control does not establish profitability.

The future September 8–October 30 holdout remains unconsumed. Neither variant
is eligible to use it. No model was promoted and no orders were placed.

## Evidence and follow-through

`artifacts/revised_research/study/study_report.json` contains the frozen
protocol and its hash, source hashes, recomputed data coverage, cost scenarios,
period outcomes, uncertainty bounds, and explicit holdout status. The adjacent
opportunity CSVs contain each signal, next-open entry, exit, and selection.
The original protocol and result artifacts remain preserved.

Regression tests cover past-only shock calculations, exclusion of the shock
from volatility, future-bar invariance, missing timestamps, zero volatility,
unknown rules, completed-family enforcement, and output preservation.

`bash scripts/agent_validate_changed.sh --skip-runtime-smoke` passed lint,
types, compilation, forbidden-pattern checks and the mapped regression suite
(details: `/tmp/revised-agent-validation.log`). The six focused tests passed
separately, and both completed protocols were checked against the final
registry to confirm they are blocked before data loading.

The separate read-only `/healthz` smoke returned HTTP 503. Its response reports
`required_model_stale` (51 days against a 14-day maximum) and
`replay_live_parity_gate_failed`; broker connectivity and database checks pass.
The latest scheduled training attempt reports `no_qualified_candidate`.
The response is retained in `/tmp/revised-research-health.json`. Service health
therefore did not pass; these research results do not resolve those operational
gates. No alerting code changed, so the previous non-sending incident check was
reused.

The economic result does not justify another automatic parameter search.
Before proposing another study, prioritize the already identified execution
evidence gaps: retained decision lineage, fee provenance, and time-aligned
broker position snapshots. These are separate from this completed research
follow-up. Historical missing evidence cannot be reconstructed by retraining.

Rollback: revert the focused runner and its new tests to remove these workflow
restrictions; preserve protocols, dispositions, and result artifacts for audit.
