# Focused research decision — September 6, 2026

**Decision: revise the research hypothesis; do not promote or keep tuning this
fixed grid.** Retire the tested 1-, 5-, and 15-minute variants. The 60-minute
variant remains inconclusive because it has only 135 validation trades against
the predeclared minimum of 250; its observed results also fail the economic
criteria. No variant qualifies for final confirmation.

## Scope and predetermined protocol

[FOCUSED_RESEARCH_PROTOCOL.json](FOCUSED_RESEARCH_PROTOCOL.json) records the
AAPL-only, long-only continuation hypothesis, 10:00–15:00 New York session,
past-only SMA/MACD conditions, next-minute-open entry assumption, four fixed
horizons, costs, acceptance criteria, and stopping rule. The signal rule and
cost grid were recorded before the new horizon outcomes. Subsequent changes
added descriptive reporting and source hashes, without changing the rule or
acceptance criteria.

The study combines two source datasets, verifies their AAPL content hashes,
resolves overlapping timestamps using the later acquisition, and recomputes
calendar coverage. It covers **285 sessions from July 21, 2025 through
September 4, 2026**, rather than the previous 126 sessions. Overlapping closes
have zero conflicts. The older parent dataset failed its overall gate because
of MSFT gaps; its AAPL subset passed. That distinction remains in the report.

Development uses 115 sessions in 2025. Retrospective validation uses 61, 62,
and 47 sessions in three subsequent periods. Their mean daily open-to-close
changes differ (-12.29, +10.35, +40.45 bps), with daily standard deviations of
143.61, 142.97, and 145.55 bps. These are distinct historical samples, not proof
of robustness to every market regime. The existing regime classifier assigns
almost all bars to sideways or volatile; directional regime coverage remains
limited.

## Results

The original model's three inner-validation folds each rejected all four
tested percentile thresholds. Six threshold groups had positive gross
markouts overwhelmed by the configured 6 bps round-trip cost; six had
nonpositive gross markouts. Gross group means ranged from -1.126 to +0.392 bps.
No threshold was relaxed and no promotion gate was changed.

The separate fixed-rule study produced these pooled retrospective validation
results. Costs below are **round-trip**; the protocol's primary assumption is
6 bps **per execution**, or 12 bps round-trip.

| Hold | Validation trades | Gross bps/trade | Net at 6 bps round-trip | Net at 12 bps round-trip | Disposition |
|---|---:|---:|---:|---:|---|
| 1 minute | 11,066 | 0.071 | -5.929 | -11.929 | Retire tested variant |
| 5 minutes | 2,190 | 0.368 | -5.632 | -11.632 | Retire tested variant |
| 15 minutes | 692 | 0.186 | -5.814 | -11.814 | Retire tested variant |
| 60 minutes | 135 | 0.093 | -5.907 | -11.907 | Inconclusive; do not promote |

All four variants fail positive net expectancy across the validation periods
at the primary assumption. They beat an always-long control primarily by
avoiding some costly trades; that relative improvement does not make their
own P&L positive. Break-even costs and all six cost scenarios, including the
previously measured quote-derived stress assumption, are retained by period.

The uncertainty calculation resamples five-session blocks, keeps strategy and
control paired, uses 10,000 seeded replicates, and adjusts eight lower bounds
(two metrics across four horizons). All candidate net lower bounds are
negative. This adjustment applies to this preregistered family; it cannot
correct for an unknown number of earlier project experiments. Accounting uses
fixed initial notional and nonoverlapping positions, not compounded portfolio
returns. Open-price simulation does not prove attainable fills, impact, or
queue position.

## Untouched data and execution reconciliation

September 8–October 30, 2026 is reserved as a future final period and has not
been consumed. Prior project datasets are not described as certified untouched
data. Because no retrospective variant qualifies, the final period remains
unopened; there is no candidate authorized to advance to it.

The execution audit reads decisions, orders, fills and TCA as hashed, stable
input snapshots. It found:

- 7,384 unique valid fills; all link to retained order records.
- Only 277 link to retained decisions; 7,107 do not.
- 6,117 link to nonpending TCA evidence; 1,267 do not.
- All 7,384 lack known fees, so net profitability is unavailable.
- 5,854 inferred FIFO entry/exit matches, explicitly distinguished from
  broker-confirmed lots or strategy exits.
- 7,384 cumulative-order comparisons use snapshots older than the associated
  last fill and are therefore not treated as quantity confirmations.
- 699 other recorded cumulative quantities differ from retained fill sums.
  Different retention histories can explain these discrepancies; they are not
  proof of a current broker-position error.

The tool supports matching opening/closing position snapshots, checks account
and paper/live identity, limits arithmetic to the snapshot interval, and
reports quantity differences. Those historical boundary snapshots are absent.
Current health cannot reconstruct old positions or missing fees. The tool
does not place orders, repair broker state, or relabel paper results as real
execution evidence.

## Reproducible artifacts and validation

Generated artifacts reside in `artifacts/focused_research/`:

- `abstention/original_h1_diagnostic_training_report.json`: threshold reasons,
  gross/net costs, scores and opportunity counts for the original model.
- `study/study_report.json`: protocol/source/code hashes, recomputed coverage,
  four-horizon study, period results, adjusted uncertainty and dispositions.
- `study/opportunities_h*.csv`: timestamped nonoverlapping opportunity ledgers.
- `execution_reconciliation.json`: joins, quantity checks, fee gaps, FIFO links
  and source stability evidence.
- `health_snapshot.json`: the non-mutating service health check.

Implementation changes for this cycle are in
`ai_trading/research/focused_cycle.py`,
`ai_trading/tools/execution_evidence_reconciliation.py`, and the existing
trainer's abstention diagnostics. Regression tests cover cost-vs-support
rejections, future-bar invariance, missing bars, session cutoffs, nonoverlap,
paired uncertainty, duplicate/conflicting fills, partial FIFO exits, fees,
order quantities, and account/time-aligned position snapshots.

`bash scripts/agent_validate_changed.sh --skip-runtime-smoke` passed lint,
type, compile, forbidden-pattern checks and **600 related tests**. The final
future-bar-invariance test was then checked with the focused research tests.
Health and the non-sending incident snapshot passed separately outside the
sandbox. Detailed validation output is in `/tmp/focused-final-validation.log`.
No trading configuration, promotion authority or service deployment changed.

## Handoff and next action

The retrospective research cycle and reconciliation audit are complete.
Remaining evidence limitations are historical decision/fee retention,
time-aligned broker position snapshots, limited directional regime coverage,
and the future final period. Do not expand this grid after seeing its results.
The next research action requires a **new economic hypothesis**, specified
before testing. Any future execution-validation effort should first capture
complete decision lineage, broker quantity snapshots and fee provenance.
Revert this cycle's source/test changes to roll back its tools; retain the
protocol and result artifacts for audit. Previous task changes and user edits
to Codex configuration and AGENTS.md are separate work.
