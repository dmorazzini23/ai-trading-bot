# September 26 economic decision and evidence repair

This is an existing-evidence review and a measurement change. It does not
register or run a trial, train a model, place an order, read the September 9–
December 8 holdout, change any trading gate, or authorize promotion. Cost
amounts below are assumptions unless identified as broker verified.

## 1. Economic diagnosis: retire the tested hypothesis

The immutable September 21 [replacement trial](../artifacts/model_replacement_20260921/report.json)
used 94,200 out-of-fold development opportunities and selected 30,802
(32.70%). It reported -3.1975 bps net per *opportunity* at a fixed 10 bps
round-trip cost. Because unselected opportunities contribute zero, the
implied mean net per *selected proxy trade* is -9.7786 bps, and mean gross
per selected proxy trade is **+0.2214 bps**. That 0.2214 bps is the
break-even round-trip cost for the observed selections. Even the 6 bps
sensitivity would be negative. All five purged folds were negative at 10 bps;
the session-bootstrap lower bound was -3.4457 bps per opportunity. These are
historical bar-open proxies, not executable fills or verified net returns.

The paired always-long baseline was -9.9472 bps net per opportunity on the
same opportunities and 10 bps assumed cost. The model's +6.7497 bps paired
improvement only means it lost less under that assumption. It does not imply
positive economics. No matched, risk-normalized IVV/MTUM benchmark was
computed by this trial, so public fund returns cannot be used as that
comparison. The tested day-model hypothesis remains rejected; adding samples
to a near-zero gross edge does not repair its cost deficit.

The consumed slower ETF trend trial is separately
`inconclusive_budget_consumed`: only four common periods and 21 selections
survived its registered complete-minute rule. Its descriptive -5.35 bps per
opportunity at assumed 10 bps cost versus +14.66 bps matched always-long is
not a qualified result. The existing replay is also below its 250-sample gate
and has not established positive net edge. None of these results supports
changing a trading threshold, overriding `required_model_stale`, or promoting
a model.

## 2. Quote and execution measurement

The September 25 seven-day scorecard read 2,414 decision rows. Two quotes
passed the one-second record-time audit; 2,412 were rejected for age over
1,000 ms. A bounded read of those same rows found a 260.6 ms median *reported*
quote age at the quote gate but an 8,956.4 ms median from quote timestamp to
decision record capture. All 2,414 rows had record-time age greater than the
reported age; 2,402 used `record_capture` as the decision timestamp basis.
The updated audit reproduced two admitted rows and 2,412 age rejections;
1,865 rows reported a fresh quote at the earlier gate but were stale by
record capture. None of the historical rows contains `quote_observed_at`.
The record reads a prior quote-gate telemetry snapshot. Thus the reported age
describes an earlier observation, while record capture happens later. It is
not evidence that the quote remained fresh when the decision was recorded.

Future decision records now retain the telemetry update time as
`quote_observed_at`, separate from the exchange quote timestamp,
`decision_ts`, and `recorded_at`. The quote audit reports counts of fresh
reported age with stale record-time age and of valid observation times. Its
one-second admission rule is unchanged. A future explicit, genuinely causal
decision timestamp and matching quote observation would be needed to prove
freshness at a trading decision; these older rows cannot be relabeled.

September 22's seven local fills map to eight broker executions because the
MSFT sell-2 local row aggregated two executions. Two fills were operational
EOD exits, without strategy decision or arrival-quote TCA records. The
same-account order identity and quantity can be reconciled, but strategy
decision/TCA completeness cannot be claimed for those exits. Broker `FILL`
activities lacked per-execution fee totals. The account-level `FEE` rows had
no order or execution identifier and cannot be allocated to fills. Missing
fee activity does not establish zero fees. Keep gross amounts, cost scenarios,
account charges, and unknown verified net P&L separate. A later natural-fill
session must supply an actual decision/quote/order/execution chain and an
authoritative complete fee source before execution economics can be verified.

## 3. Candidate for separate review, not an experiment

One genuinely different mechanism worth *specifying*, subject to a data
preflight, is weekly **relative strength across the six already governed
ETFs** (DIA, IWM, QQQ, SPY, XLE, XLF), rather than the rejected daily
single-stock classifier or the consumed absolute 60-session ETF trend rule.
At each completed Friday close, rank the six by trailing 20-session
distribution-adjusted return divided by trailing 20-session realized
volatility. A frozen design could hold the two highest ranked ETFs for five
regular sessions, entering at the next executable open and allowing cash
when the market filter is negative. This is a proposal for an economic
mechanism, **not a backtest result or a registered campaign**. Rank ties,
the exact filter, sizing, turnover accounting, split/distribution treatment,
execution assumptions, costs, and stopping rules must be frozen before a
trial is approved.

Preflight must first verify, without reading outcomes, complete timestamped
20-session lookbacks and actual entry/exit open observations on shared dates
for all six ETFs; stable source hashes; corporate-action and distribution
coverage; and enough common weekly dates for the new campaign's support
thresholds. No interpolation of missing observations or silent ticker drops.
The prior slower trial showed 88 potential five-session periods but only four
under its complete-minute DIA rule. A new data rule may use only the
observations this *different* design needs, but cannot amend or rerun that
consumed trial. If the preflight fails, the candidate stays untested.
The currently governed ETF campaign declares `adjustment: split` and explicitly
notes omitted distributions, so its bars alone cannot supply the proposed
total-return rank or dividend-aware baseline. An existing paginated
`artifacts/endpoint_proposal/corporate_actions_expanded.json` capture has 64
positive, distinct-ID cash-dividend events within 2024–2025 across the six
ETFs (24 DIA and eight for each other ETF), plus one XLE split. The earlier
narrower capture has only 62 in-window dividends, including seven SPY events;
the expanded capture includes the December 2025 SPY event. This is a useful
source candidate, not yet a validated adjustment join or proof that all
distributions are complete. Until its provenance, event-date semantics and
join to the governed bars pass a separate preflight, data support for the new
hypothesis remains **unverified**. No data was purchased or added.

If separately approved, a new immutable one-trial contract would use only
governed 2024–2025 development data; the reserved holdout remains untouched.
It must compare net results on identical opportunities to cash and matched
always-long ETF exposure, report gross break-even cost and fixed 6/10/20 bps
round-trip scenarios, use purged time splits/session bootstrap, and account
for realistic capital, overlap, turnover and risk. It must pass existing
provenance, cost, sample, replay and promotion gates. An inconclusive or
negative result consumes its budget and cannot be tuned in place.

## 4. Promotion and acceptance

No model or trading setting changes now. The next evidence sequence is:
data-support preflight; separately approved frozen trial; positive and robust
development economics versus matched baselines; replay qualification;
natural paper fills with causal quote and complete execution-cost evidence;
then existing broker/accounting, recovery, risk and live-canary gates. A
healthy process or a larger sample alone is not a positive return claim.

Acceptance for this task: the economic calculations reproduce from the
immutable report; future decision rows preserve the distinct quote observation
time; audit tests show a delayed record still fails one-second freshness;
and no order, training run, search, budget reset or holdout evaluation occurs.
The evidence gaps and future approval point remain explicit.
