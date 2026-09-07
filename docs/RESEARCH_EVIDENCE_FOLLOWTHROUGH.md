# Research evidence follow-through — September 7, 2026

The four-item workflow is implemented. Observed session and cost validation
remain pending sufficient contemporaneous evidence. The serving decision
remains abstain; no strategy or allocation policy was promoted.

## 1. Complete-session audit

The daily research workflow now runs `ai_trading.tools.paper_evidence_review`
after its replay generation. It selects the latest completed NYSE session,
including holiday and early-close handling, and waits until 15 minutes after
the close. The installed daily timer is active and invokes the existing
repository script in a fresh process, so this step requires no daemon restart.
Its next scheduled invocation was verified as September 7 at 20:37 UTC.
That holiday run will still inspect September 4; September 8 is the next
eligible session to collect a new complete trading-session record.

The step consumes captured position boundaries, decisions, orders, fills and
TCA. It writes a run artifact, copies
`paper_evidence_review_latest.json` into the research report root's `latest`
directory, and adds a compact result to the existing operator summary.
The step does not place orders or send messages. Missing or unstable input
files, missing boundaries, no execution samples, and unverified fees stay
explicitly pending. Existing research execution continues through blocked
individual model steps, so an unqualified model does not skip this audit.

The current September 4 audit reports missing boundary pairs, no execution
samples, and unconfirmed positions. Subsequent boundary capture is running,
but these later observations cannot reconstruct September 4. A zero-fill
abstaining session is not a successful execution validation.

## 2. Cap-selection diagnosis

The fixed diagnostic preserves all 4,349 recorded input rows, seed 42, the
$25,000 symbol cap and $150,000 gross cap. It compares recorded order sequence
with reversed order sequences only within identical timestamps. No later
decision is moved earlier. There are 1,516 timestamp groups with ties and no
recorded pre-trade edge scores suitable for quality-ranked allocation.

The initial diagnostic found a 0.716 bps sensitivity despite unchanged order
counts. The replay loop supplied only one symbol's current price when draining
fills for every symbol; other symbols could fall back to order prices. The
loop now makes all prices observed at that same timestamp available before
processing fills. Conflicting prices for one symbol at one timestamp are
rejected instead of resolved by incidental input order. No future timestamp
is consulted.

After correction, both timestamp orderings produce exactly -17.895 bps mean
candidate markout, 3,210 orders, 3,014 markouts, 1,922 adjustments, and zero
OMS invariant violations. Ordering sensitivity is zero. This fixes simulation
measurement; it does not establish a better allocation policy.

The uncapped baseline is -17.276 bps and the capped candidate -17.895 bps.
The remaining **-0.618 bps** difference is attributed to the retained order set;
fill and cost differences contribute zero and the attribution residual is
zero. This supersedes the earlier -1.696 bps measurement, which included the
timestamp-price artifact. The gate remains failed and the caps remain intact.

| Symbol | Retained baseline markout | Removed baseline markout | Difference, bps |
|---|---:|---:|---:|
| AAPL | -20.266 | -17.345 | -2.921 |
| AMZN | -19.713 | -19.504 | -0.209 |
| MSFT | -13.989 | -9.848 | -4.141 |

These are descriptive markout comparisons, not realized P&L or available
pre-trade rankings. Selecting the removed orders using these outcomes would
introduce hindsight. No new ranking rule is warranted by this evidence.

## 3. Observed execution-cost comparison

The daily review pairs simulated and observed fills by client-order identity,
paper account, symbol, side and an identical arrival-price benchmark. It
deduplicates fills, quarantines conflicts, aggregates partial fills by quantity,
and reports signed slippage error and the 90th percentile absolute error.
Simulated and observed quantities are retained when they differ.

The predeclared comparison-support threshold is 30 paired orders over five
sessions. This is a minimum for describing discrepancies, not proof of
profitability. Observed fees require broker provenance; absent simulated and
observed fee contracts keep net-cost validation unavailable. Paper fills do
not establish queue position or market impact.

The current audit has **zero eligible pairs**: all 7,384 historical fills lack
the required verified paper-account identity. No cost calibration is changed
from these records. Future daily reviews will recompute the comparison when
new identified fills and matching replay evidence exist.

## 4. Feasibility before another study

New focused studies now require a feasibility review bound to their exact
protocol hash and the review artifact's SHA-256. Missing, blocked, modified or
mismatched reviews stop the study before data loading and evaluation. Existing
retirement and no-overwrite rules remain in force.

`ai_trading.tools.research_feasibility` screens development-period opportunity
ledgers, using five-session bootstrap blocks and 5,000 seeded replicates.
It estimates a conservative projected validation trade count and an optimistic
upper bound on gross edge. Trade support must meet the frozen acceptance
criterion and optimistic gross edge must exceed round-trip costs. A recorded
hypothesis and stopping rule are mandatory. This is an early rejection screen,
not confirmation of net edge; it assumes development frequency persists.

| Completed variant | Development-only screening result |
|---|---|
| Continuation, 1 minute | Optimistic gross edge approximately 0 bps, below 12 bps costs |
| Continuation, 5 minutes | Optimistic gross edge 0.435 bps, below costs |
| Continuation, 15 minutes | Optimistic gross edge 1.566 bps, below costs |
| Continuation, 60 minutes | Conservative projected support 115 trades, below 250 |
| Reversal, 15 minutes | Conservative projected support 22 trades, below 250 |
| Reversal, 60 minutes | Only five development trades; insufficient support |

All six are blocked by the new screen without using validation returns.
Applying the screen now is a retrospective demonstration, not a claim it was
preregistered before those old experiments. No failed experiment was reopened
and the reserved future holdout remains unconsumed.

For a genuinely new focused protocol, generate the development opportunity
ledger without evaluation-period outcomes, run the feasibility CLI with
`--protocol`, `--ledger-dir`, and `--output`, then attach an absolute review
path and its SHA-256 under `feasibility_review`. Changing the hypothesis or
acceptance criteria invalidates that review. This gate governs the focused
study runner; it does not silently alter other established training workflows.

## Artifacts, validation and remaining evidence

`artifacts/four_item_followthrough/` contains:

- `paper_evidence_review.json`: diagnostic before the timestamp-price fix.
- `corrected_evidence_review.json`: current session audit, cap diagnosis,
  paired-cost rejection counts and corrected ordering test.
- `governance_summary.json` and `replay/replay_hash_20260907.json`: corrected
  fixed-input replay and preserved failed governance decision.
- `continuation_feasibility.json` and `reversal_feasibility.json`: development
  screening results and source hashes.
- `health.json`: read-only running-service health response.

Tests cover calendar boundaries, cost pairing and duplicate conflicts,
timestamp-order invariance, conflicting prices, development-only screening,
cost/support rejection, protocol binding, retirement enforcement, and daily
workflow wiring. Validation logs are `/tmp/four-items-final-validation.log`
and `/tmp/four-items-final-targeted.log`. No alert is sent by the incident
snapshot smoke check.

`bash scripts/agent_validate_changed.sh --skip-runtime-smoke` passed 952 tests
in 170 seconds, lint, type checks over 39 source files, compilation and
forbidden-pattern checks. The final operator-summary wiring additionally
passed 28 targeted tests, lint, mypy and compilation. The separate health smoke
returned HTTP 503 with the existing replay/stale-model flags; broker status is
connected and the model decision is abstain. The non-sending incident check
passed.

The outstanding work requires new evidence: a complete observed paper session,
verified fees and enough matched executions. Health remains degraded by the
stale-model and replay gates. Do not lower the gates or initiate trades merely
to finish these checks. The scheduled review will retain the pending reasons.

Rollback: revert this task's replay timestamp-price handling, new review and
feasibility modules, focused-runner gate, and research automation wiring.
Preserve the evidence artifacts and earlier task changes. Scheduled research
jobs load these modules in fresh processes; live trading controls are unchanged.
