# Research reset: September 8–October 8, 2026

The objective is a defensible decision about one trading edge after costs.
Profitability, a newer model timestamp, and additional features are not completion
criteria. The existing paper-trading and model-promotion gates remain authoritative.

## Operating policy

`config/research_reset.json` redirects daily, weekly, monthly and weekend research
automation to three required steps: paper-account reconciliation, completed-session
evidence review, and the research reset scorecard. Manual strategy-change plans
follow the same restriction. Incident replay and existing manual authority reviews
remain governed by their existing checks. Missing or malformed reset policy blocks
automation; the October 8 review date does not automatically restart broad search.
Resuming requires an explicit review and a deliberate policy change.

Discretionary features, new models, and unregistered parameter searches are paused.
Correctness fixes and work that resolves a named evidence blocker remain in scope.
This policy controls the canonical research orchestrator, not arbitrary commands
an operator runs directly. Existing health, capture, and governance timers continue.
The research timers read this checkout on each invocation; no trading-service
restart is needed. Notifications retain existing operator configuration; the
validation runs for this task call the Python orchestrator directly and send none.

## One fixed research question

Does a simple, slower ETF trend filter add value over cash and identical always-long
opportunities after fixed costs? `config/slower_horizon_campaign.json` freezes the
rule, development interval, symbols, execution convention, support thresholds,
cost scenarios, bootstrap, and rejection criteria before evaluation.

The rule observes the previous completed session close relative to 60 exchange
sessions earlier, enters at the regular-session open on a fixed five-session grid,
and exits at the open five sessions later. No overlapping position per symbol,
shorting, leverage, compounding, optimization, or model fitting is introduced.
Every control uses the same complete opportunities and equal-notional capital
slots. Inactive strategy slots remain cash with zero assumed interest. Missing
sessions exclude an opportunity; they never compress the lookback or move its exit.

The economic hypothesis is gradual adjustment to information. General time-series
momentum research motivates testing persistence, but does not validate this ETF
rule or these horizons: [original research and data](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data).
Whipsaw, passive market exposure, overnight gaps, fees and distributions can erase
or explain apparent gains.

The development interval is 2024–2025, using the existing governed SIP acquisition.
It was used in earlier research and is not untouched. September 9–December 8, 2026
remains reserved. The evaluator checks acquisition metadata before opening price
files and rejects any other date interval. Current operational evidence checks do
not evaluate holdout strategy performance.

The canonical campaign ledger is `artifacts/research_reset/campaign_state.json`.
Changing the output directory does not reset its one-trial budget. Invalid data
blocks before claim; an inconclusive outcome consumes the trial. A crash after
claim requires an audit. A supported failure retires the hypothesis. A passing
result permits planning only and never changes model eligibility or trading authority.

```bash
./venv/bin/python -m ai_trading.tools.slower_horizon_study \
  --acquisition artifacts/research_foundation/acquisition_sip_repaired.json
```

The primary round-trip cost is 10 bps; 6 and 20 bps are fixed sensitivities, not
choices for rescuing a failed result. Price markouts omit distributions and cash
interest. They are not total portfolio returns, CAGR, Sharpe, or drawdown. An
economic decision would additionally require distribution-adjusted returns, a
capital-constrained portfolio replay, and independently justified execution costs.
ML must later beat the frozen simple baseline under the same information and
execution assumptions; this reset fits no model and claims no ML advantage.

## Weekly evidence scorecard

The canonical CLI is `ai_trading.tools.research_reset`. Every scheduled evidence
run writes its own scorecard and updates
`research_reports/latest/research_reset_scorecard_latest.json`. Weekly runs provide
the weekly cadence; daily snapshots provide diagnostics and are never counted as
additional experiments. Source hashes, read stability, exclusions and timestamps
are preserved. Missing or old sources remain visible.

| Measure | Definition and interpretation |
| --- | --- |
| Concluded experiments | Unique campaign trials completed in the trailing seven days, verified against the immutable contract and report hashes. Inconclusive outcomes are distinguished from retirement and success. |
| Untouched net performance | Unavailable until a separately governed untouched evaluation; development results are never substituted. |
| Cost sensitivity | Fixed 6/10/20 bps scenarios and gross break-even cost, on the same opportunities. Assumed costs are not measured fees. |
| Concentration | Quarterly and symbol results plus the slower study's largest symbol share of absolute net contributions. No unsupported time trend is inferred from one snapshot. |
| Execution completeness | Complete causal chains divided by identifiable unique observed fills in the trailing seven days. Missing identifiers/conflicts make the fraction unavailable. Zero fills also means unavailable, never 100%. |

A complete chain requires an explicit decision time after the source minute has
closed, a valid quote no more than one second before the decision, an account-scoped
order with the same order identity between decision and fill, valid fill values,
and broker-sourced USD per-fill total fees. Account-level fee records are never
allocated to guessed fills. This is evidence coverage, not independent certification
of broker completeness or round-trip profit. Paper trading also excludes market
impact and queue-position effects: [Alpaca limitations](https://docs.alpaca.markets/us/docs/paper-trading).

## Thirty-day working sequence

1. **September 8–14:** reconcile existing artifacts, implement and verify the reset,
   complete the one registered development trial, record the highest-value blocker.
2. **September 15–21:** review the next weekly scorecard. Investigate only new or
   unresolved causal-link and accounting gaps. No forced trades to create samples.
3. **September 22–28:** review stability and concentration of the frozen evidence;
   obtain authoritative fee records if available. Preserve failed and inconclusive
   experiments without parameter tweaks.
4. **September 29–October 8:** decide whether evidence supports further investment,
   a separately justified campaign, or continuing the pause. The reserved holdout
   extends beyond this review; thirty days does not complete that evaluation.

For unavailable per-fill total fees, the required input is a broker-confirmed
execution identifier, account, currency, quantity, timestamp and complete total
charge, with an explicit completeness basis. The configured paper Trading API
currently does not establish that contract. This gap cannot be repaired by
assuming zero fees, spreading aggregate charges across fills, or creating orders.

## Validation and rollback

### September 9 evidence-gap follow-through

The timestamp-only sampling audit is now available independently of any strategy
evaluation and runs before price analysis or budget claim for an unclaimed slower
study. The completed campaign still exits immediately as budget exhausted. This
new preflight cannot reopen it. The audit checks source hashes, stable reads,
timestamp uniqueness/timezones, the fixed development interval, and the exact
lookback/holding grid. A passing count is only an upper bound, never eligibility.

```bash
./venv/bin/python -m ai_trading.tools.research_feasibility \
  --protocol config/slower_horizon_campaign.json \
  --sampling-acquisition artifacts/research_foundation/acquisition_sip_repaired.json \
  --output artifacts/research_reset/sampling_support.json
```

`sampling_support.json` confirms 88 potential five-session periods before common
coverage filtering. DIA permits four under the registered complete-session rule;
each other ETF permits 88. Its 973 missing minutes occur in 243 sessions, but **none
is an opening or closing minute**. Shared support is therefore at most 24 selections,
below the fixed 100-selection and 60-period requirements, without inspecting a
signal or return. This structural impossibility can now be detected before a trial.

`dia_provider_recheck.json` records a bounded same-feed requery of the first,
middle and last gap sessions: January 12, 2024; November 4, 2024; and August 28,
2025. All three returned the identical missing minutes; zero omissions recovered.
The original dataset was not modified. Alpaca explains that absent eligible trades
can produce no bar: [Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq).
This is a possible explanation, not an independent determination of each gap's
cause. The three-session sample is not proof about every omitted minute.

`fee_followthrough.json` audits the actual available activity schema: all 446 fill
activity rows lack `fee_amount`; all 81 fee activity rows lack an order or fill
reference. The snapshot was fetched September 9 at 00:39 UTC and was reused for
this schema check; it is not a new account fetch. Its independent raw-row grain
must not be confused with 400 reconciled local records. No fee amount, execution
link, or total-charge completeness is inferred. See the official
[account activities contract](https://docs.alpaca.markets/us/docs/account-activities).

The scheduled scorecard now includes both diagnostics. A current local snapshot
is `artifacts/research_reset/followthrough_scorecard.json`; its highest-value
blocker is `sampling_support_insufficient`. Future snapshots expose missing/stale
support audits rather than treating old counts as fresh evidence.

Follow-through validation: the changed-file validator passed lint, mypy, compile
and 111 selected tests. Subsequent scorecard and preflight refinements passed 14
and 11 focused tests respectively, with separate lint/mypy/compile. Host health
returned the existing HTTP 503 readiness flags; the non-sending incident snapshot
passed. Documentation checks passed. The original single trial and its report hash
were verified unchanged. No new strategy outcome was calculated.

The concrete candidate for a **future, separately justified data contract** is to
require only the completed closes and executable entry/exit observations used by
the fixed rule, with explicit adjustment/distribution handling, valid prices and
causal timing. No interpolation or invented bars are justified. This observation
does not authorize changing the completed experiment, dropping DIA, running a
new variant, or evaluating the reserved holdout. Support counts must pass before
any separately authorized new campaign spends a trial.

September 9 completion evidence is saved in `artifacts/research_reset/`:

- `report.html`: portable readiness report; `artifact.json` is its source input.
- `readiness_scorecard.json`: frozen copy of the verified scheduled scorecard.
- `scheduled_verification.json`: workflow executed and operator report verified.
- `evidence_checks.ipynb`: executed assertions against saved evidence and hashes.
- `study/report.json`: immutable trial result, `inconclusive_budget_consumed`.

The slower study retained four common periods, 24 opportunities and 21 selections.
DIA had 243 incomplete sessions; requiring complete sessions across each lookback
and hold excluded 84 candidate periods for DIA. The descriptive result was -5.35
bps per common opportunity at 10 bps assumed cost, versus +14.66 for the matched
always-long control. This sample does not support a strategy conclusion.

Audit correction: four periods equal one bootstrap block. The original report's
near-zero-width interval is degenerate, not evidence of precision. The immutable
report and its hash remain unchanged; the scorecard suppresses this interval and
the evaluator now reports uncertainty unavailable below two blocks. Regression
tests cover both paths. The development trial was not rerun after this correction.

The fresh 90-day accounting snapshot contains 527 activities, 400 matching order
quantities and 400 local records without verified per-fill total fees. The weekly
source window contains 1,940 decision rows, zero order rows and zero fill rows.
All three source reads were stable. Two campaign completions are hash-verified;
neither is evidence of a tradable edge. Future work must first assess usable
opportunity support, not rely on aggregate minute-bar coverage.

`bash scripts/agent_validate_changed.sh --market-hours --skip-runtime-smoke`
passed lint, mypy, compile and 101 selected tests. A subsequent account/order-side
check passed lint, mypy, compile and 13 focused tests. The earlier non-skipping
validator reached the sandbox network boundary; host checks independently passed:
health returned HTTP 503 with `required_model_stale` and
`replay_live_parity_gate_failed`, and the non-sending incident snapshot assertions
passed. The completed evidence-only automation run is
`research-reset-20260909-validated`; all three steps returned zero and its operator
report verified. No orders, model training, promotion or task-generated messages
were sent by that run.

The analytical report passed canonical artifact validation and portable payload
verification. The MCP renderer returned a payload without an inspectable hosted
resource in this Codex surface, so final delivery uses HTML. No compatible Chromium
was installed: browser layout/source-dialog QA was unavailable, and the portable
builder verified payload equality and its semantic fallback structurally. Notebook
assertions and `git diff --check` passed.

Targeted tests cover causal ordering, account/order identity, fee provenance,
missing data, immutable result verification, duplicate grain, fixed entry/exit
timing, paired baseline rejection, scheduler reset plans and holdout rejection
before price loading. The repository changed-file validator supplies lint, typing,
compilation and regression checks. Host health and non-sending incident checks
are required separately when sandbox networking cannot reach the service.

Rollback removes the reset routing and its dedicated tools/tests while preserving
campaign ledgers and reports as evidence. Explicitly reviewing and disabling the
policy restores existing orchestration; it does not restore a retired experiment's
trial budget or grant trading authority. Preserve unrelated user changes.
