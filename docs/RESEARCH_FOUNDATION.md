# Research foundation: cost audit, broader data and a bounded campaign

September 8, 2026. This work addresses the five requested priorities. Research
outputs never grant trading authority. The prior momentum studies remain retired.

## 1. Execution-cost evidence

`research_foundation` audits seven calendar days of decision quotes, rejects
invalid/old/stale observations, deduplicates quotes and groups observed spreads
by symbol, recorded liquidity regime and intended order notional. Size buckets
are <=$1,000, $1,000-$10,000, $10,000-$25,000 and >$25,000. They describe intended
size, not displayed depth or measured market impact. Quote support requires 30
observations per bucket; that threshold does not certify accuracy.

The refreshed 14:22 UTC audit retained 1,310 quotes in 99 buckets. It rejected
17,593 rows without valid quotes, 10,579 outside the window and 624 older than 1,000 ms.
All retained observations lack verified alignment to the recorded decision time.
The recorded decision timestamp can refer to an earlier bar; a fresh quote at
capture is not proof that it was available at that bar's timestamp. Thus these
are spread observations, not causal replay execution prices.

Examples for normal liquidity and intended $1,000-$10,000 orders:

| Symbol | Samples | Median full spread bps | P90 full spread bps |
| --- | ---: | ---: | ---: |
| AAPL | 96 | 1.84 | 16.57 |
| AMZN | 64 | 2.73 | 16.24 |
| MSFT | 42 | 17.09 | 88.28 |

The artifact adds 4 bps slippage plus 2 bps assumed fees to each observed spread
scenario, once per round trip. These buffers are assumptions; wide valid spreads
are retained. No cost estimate is transferred from these stocks to the new ETFs.

The fresh paper-account review retrieved 537 activities: 454 fills and 83 fees.
408 historical order quantities match; all 408 linked records still lack verified
per-fill total fees. Aggregate fee records remain unallocated. The configured
Trading API has not provided transaction confirmations from the separate Broker
document API. Existing execution comparison has zero supported pairs. Therefore
realistic total execution-cost calibration remains **blocked by evidence**.
Paper trading also omits impact and queue-position effects, per
[Alpaca's limitations](https://docs.alpaca.markets/us/docs/paper-trading).

## 2. Price, label and acquisition audit

All three original source CSV hashes match their manifests. No invalid OHLC rows,
duplicate/nonmonotonic timestamps or >20% overnight discontinuities were found.
The original feed is IEX with raw prices. A discontinuity screen is not independent
corporate-action verification. New acquisition requests split-adjusted prices;
the new study holds intraday and uses normalized prior ranges, not dividend-inclusive
overnight returns. It makes no total-return or point-in-time universe claim.

Alpaca minute bars are labeled at the interval start. Existing training labels
are same-close to future-close research targets; they are not executable returns.
The previous momentum study uses next-open entry. The new study allows a full
minute after the signal closes, then uses the next bar open for entry and an
open 120 minutes later for exit. Tests perturb future prices to verify that
the selected signal and its threshold do not change. These conventions follow
[Alpaca's bar documentation](https://alpaca.markets/learn/stock-minute-bars) and
[adjustment contract](https://docs.alpaca.markets/us/reference/stockbarsingle-1).

The audit found and fixed a concrete acquisition defect: Alpaca-py's `limit`
caps total returned rows across SDK pagination. Extended-hours observations could
exhaust 10,000 rows before a 20-session window ended. The canonical fetcher now
advances its timestamp cursor until the window is exhausted. Capped old checkpoints
are refreshed once; completed pagination is recorded. Regression coverage exercises
multi-page acquisition and old-checkpoint repair. Quality gates were not relaxed.

## 3. A distinct economic hypothesis

Registered hypothesis: `etf_opening_shock_reversal_v1`. Large opening declines
may contain temporary liquidity pressure that partially reverses. This is inspired
by [research on short-term reversals](https://www.newyorkfed.org/research/staff_reports/sr513.html),
but applying that mechanism to this intraday ETF rule is an untested inference.
It can fail because the decline conveys information, costs overwhelm recovery,
or the proxy does not identify liquidity shocks.

The fixed rule buys after the first 30-minute return is <= minus half the median
normalized daily high-low range over 20 prior complete sessions. Entry occurs
31 minutes after the session open, exit 120 minutes later. One opportunity per
symbol/session; no fitting or parameter search. Primary round-trip cost is 10 bps,
with 6 and 20 bps sensitivities. These are conservative scenarios, not calibrated
ETF costs. Cash and always-long use identical eligible opportunities.

Acceptance requires eight supported quarter folds (>=30 selections each), positive
net edge and a positive 95% whole-session bootstrap lower bound at 10 bps, six
profitable quarters, four profitable symbols, and outperformance versus always-long.
No account return curve is constructed from these markouts.

## 4. Broader development coverage and reserved evidence

The universe is DIA, IWM, QQQ, SPY, XLE and XLF over January 2024-December 2025:
502 expected sessions and 194,700 expected regular-session minute bars per symbol.
The ETFs broaden exposures beyond three technology stocks but remain correlated;
they do not represent all markets or an unbiased point-in-time equity universe.

Initial IEX acquisition failed completeness for every symbol. A SIP comparison
revealed truncated windows and led to the pagination correction. Earlier incomplete
acquisitions are preserved. The feed was amended from IEX to SIP before evaluating
any strategy outcome; the campaign state preserves the original contract and reason.
Thresholds, universe, dates and budget did not change.

Corrected SIP acquisition passed: IWM, QQQ, SPY, XLE and XLF each have all
194,700 expected bars. DIA has 193,727 (0.5% missing), within the unchanged 2%
limit. Total coverage is 1,167,227 bars. Prior incomplete acquisitions remain
separate artifacts; their apparent gaps must not be interpreted as strategy results.

The new runner requires a verified acquisition hash, exact development interval,
feed/adjustment match and recomputed coverage/regime evidence. It excludes incomplete
sessions without interpolation. September 9-December 8, 2026 is reserved prospectively;
the development runner rejects an acquisition extending into that period. Those
future observations cannot be validated now. No automatic holdout evaluation exists.

## 5. Enforced research budget

`config/research_foundation_campaign.json` registers one hypothesis and one trial.
The existing experiment-ledger module now supports immutable campaign registration,
locked budget claims before outcomes, and immutable completion with a report hash.
Changing output directories or evidence signatures cannot refund a claimed trial
when the same campaign ledger is used. This is workflow enforcement, not protection
against deliberately replacing the ledger or source code.

Invalid data blocks before claiming. A conclusive failure retires the hypothesis;
insufficient statistical support consumes the single trial and is inconclusive.
A crash after claim requires explicit audit, not an automatic retry. Passing only
permits planning an untouched evaluation. The ledger does not bypass existing
registry gates. Bounded testing addresses the repeated-search risk discussed in
[backtest-overfitting research](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

## Reproduction and artifacts

Audit: `ai_trading.tools.research_foundation`.
Acquisition: `ai_trading.tools.historical_training_backfill`.
Study: `ai_trading.tools.opening_reversal_study --campaign-state
artifacts/research_foundation/campaign_state.json --acquisition
artifacts/research_foundation/acquisition_sip_repaired.json --output-dir
artifacts/research_foundation/reversal_study`.

Artifacts under `artifacts/research_foundation/` preserve quote audit, rolling cost
diagnostics, fresh broker activities/accounting, both incomplete acquisitions,
corrected acquisition, campaign registration/amendment and study outputs.

The new code is research-only; the shared acquisition fix also affects scheduled
research. No service restart, trading configuration change or alert was required.
Rollback reverts the changed acquisition/ledger modules and new research tools,
preserving evidence. Reverting pagination reintroduces the identified coverage defect.

## Actual study result and validation

The registered study completed on September 8. It evaluated 2,649 common
opportunities and selected 193. At the predeclared 10 bps round-trip cost:

- Net per common opportunity: -0.0195 bps.
- Net per selected opportunity: -0.2678 bps.
- Gross break-even cost: 9.7322 bps per selection.
- Whole-session bootstrap 95% interval: [-0.7911, +0.7563] bps per opportunity.
- Always-long net on identical opportunities: -10.9300 bps; cash: 0 bps.
- Fixed 6/10/20 bps stress net per opportunity: +0.2719/-0.0195/-0.7481 bps.

The eight supported-fold condition failed, as did positive aggregate net,
positive lower confidence bound and six profitable quarters. The outcome is
`inconclusive_budget_consumed`, not evidence of a tradable edge or a conclusive
retirement based on adequate support. The single trial is consumed; automatic
retries are blocked even with a new evidence signature. Its completed ledger
entry contains a verified report hash. No holdout was evaluated and nothing
was promoted. Any further research requires a separately justified new campaign;
changing costs to the profitable sensitivity does not make this experiment pass.

Validation: `bash scripts/agent_validate_changed.sh --skip-runtime-smoke` passed
lint, mypy, compilation and 22 targeted tests. A separate existing training/ledger
regression run passed 36 tests. Documentation validation and `git diff --check`
passed. The actual corrected acquisition and study both completed. The read-only
runtime health smoke returned HTTP 503 with the existing `required_model_stale`
and `replay_live_parity_gate_failed` attention flags; these gates remain active.

Outstanding evidence is explicit: verified per-fill total fees, causal execution
comparisons, independent corporate-action verification for older raw-price studies,
and future holdout observations. Implementing the five workstreams does not make
these externally dependent measurements complete.
