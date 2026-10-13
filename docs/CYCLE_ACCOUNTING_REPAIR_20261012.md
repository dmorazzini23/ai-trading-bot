# October 12 cycle and accounting repair

## WORKLOG

Scope: reduce established redundant broker reads and fix the October 8 AMZN
entry TCA and MSFT EOD decision report gaps. Preserve account/order/fill identity,
causal timestamps, protective exits, consumed budgets, September 9–December 8
holdout and all model/replay/cost/freshness/provenance/promotion gates.

### Slow cycles

October 12's first critical cycle took 194.75 seconds and performed six complete
history scans, each covering about 39,467 rows in 79 `all` pages. Individual
reads took 14.9–25.9 seconds. The 110-second critical threshold is unchanged.

The symbol adapter filtered after an account-wide scan; it now sends the explicit
symbol filter to both broker queries and every page. Unknown/old submission
times, nonempty `open` responses and incomplete/tied pagination retain existing
discovery/failure behavior. Unscoped risk and broker synchronization still read
the complete account. Known-intent reconciliation uses exact broker/client ID
lookups when both are available, retaining ambiguous claims on failures and
falling back to existing full-list behavior for other broker capabilities.

The final preclaim pressure check may consume this attempt's complete capacity
observation only for the same broker and within one second. Age is checked again
when the pressure guard actually consumes it, including time spent computing the
go/no-go report. Missing, stale, future or nonfinite observation times require a
fresh complete read. No persistent discovery cache or temporal history cutoff.

Read-only after-close comparison (not a market-hours cycle measurement):

| Read | HTTP requests | Observed seconds |
| --- | ---: | ---: |
| Existing whole-account snapshot | 80 | 3.8493 |
| Complete AMZN-scoped snapshot | 6 | 0.3246 |
| Known MSFT native order lookup | 1 | 0.0135 |

**Status:** implemented and locally validated; CI/deployment pending. Acceptance
after deployment: observe ordinary market cycles and the existing critical SLO,
including an ordinary EOD reduction, with fewer historical scans and consistent
broker/OMS state. Provider latency and unrelated compute costs are not eliminated
or certified by this comparison. Never force a trade for measurement.

### AMZN TCA

The request at 19:17:38 UTC October 8 was compacted at 19:18:04 by
`fill_event_fallback_match`, before its actual 19:18:12 fill. Similar symbol,
side, quantity and a broad time window are not order identity. Compaction changed
`pending_event` to false, preventing later fill reconciliation.

Fuzzy matches now retain unresolved requests. An older compacted unfilled request
can append a receipt only with matching native order and client IDs, matching
account, nonconflicting mode, symbol/side, causal aware timestamps and bounded
observed quantity. The fill's explicit mode may supply an absent original mode;
a conflicting original mode cannot be replaced. Original rows remain unchanged.
Recovered receipts carry actual fill and recording times separately; complete
fees and independently verified latency remain unknown, and promotion is false.

**Status:** established ingestion/maintenance defect fixed and validated on a
copy of the actual records. Original runtime TCA has not yet been changed.
After passing release checks, append the supported receipt through the canonical
reconciler and recheck the dated report. No estimated allocation or invented fill.

### MSFT EOD decision

The causal operational EXIT decision already exists in immutable `decision_events`
with the same native trace/account and capture before the fill. The session report
read only ordinary `decision_records.jsonl`, omitting this authoritative source.
The report now consumes the existing coherent read-only OMS diagnostic. It checks
both causal decision time and actual recording time against the fill and excludes
conflicting accounts. It does not reconstruct a decision or label EXIT as model
acceptance. Unavailable OMS evidence remains an explicit source gap.

**Status:** report omission fixed and locally verified using the actual durable
decision. Acceptance: regenerated October 8 report matches that existing decision
without writing a synthetic ordinary decision row.

## PATCHSET

- `core/alpaca_client.py`, `broker/adapters.py`: complete scoped pagination.
- `execution/engine.py`: exact known-intent reconciliation reads.
- `execution/live_trading.py`: owned bounded observations and actual fill scope.
- `analytics/tca.py`: reject fuzzy compaction; append supported recovery receipts.
- `tools/paper_evidence_review.py`, `tools/execution_evidence_reconciliation.py`:
  durable causal decisions in the session audit.

## REGRESSION COVERAGE / VALIDATION

43 initial focused tests and 164 expanded engine/restart/accounting tests passed.
Final `bash scripts/agent_validate_changed.sh` passed 93 mapped tests, Ruff, mypy
on 12 paths, compilation, forbidden-pattern checks, live health and non-sending
incident smoke. Tests include late-page/old-timestamp pending orders, outside
universe account orders, same-broker ownership, consumption-time age fallback,
fresh/stale ambiguous claims, native/mode/account/quantity/time conflicts, original
row preservation and genuine operational decisions versus late reconstructions.
Known native identity conflicts cannot fall back to a guessed symbol/quantity
join. Retrospective request and fill times require actual timezone offsets.
Exact-tip full CI remains required before deployment. No new research experiment,
training, parameter search, ticker expansion or holdout evaluation was run.

## RUNTIME CHECK / EVIDENCE

The paper service remains on clean CI-tested `1853464dc`, active with zero automatic
restarts. Health is fresh/flat and OMS checks pass; stale-model/replay flags remain.
On copies of both actual October 8 records, decision/order/TCA matches improve
from 1/2/1 to 2/2/2; both fee totals remain unknown. No original record was rewritten
and no order or notification was sent. Evidence and validation logs are retained
privately under `artifacts/maintenance/cycle_accounting_repair_20261012/` in the
runtime checkout. Existing accounting history gaps remain separate blockers.

## RISK & ROLLBACK

Broker response completeness remains mandatory; exact lookup failure cannot prove
terminal state or authorize resubmission. Operational latency after deployment
requires observation during the next ordinary session. Model qualification and
verified net economics remain unresolved. Keep source `1853464dc` and a persistent
copy of its release specification before staging. Roll back source/spec after a
closed-market exposure review if checks fail. Preserve any appended factual receipt
and all original evidence; do not erase a supported correction when reverting code.
