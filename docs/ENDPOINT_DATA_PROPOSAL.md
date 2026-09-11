# Required-observation data contract and feasibility review

Follow-through completed September 9: see `docs/ENDPOINT_PREPARATION_RESULT.md`.
Adjusted-data acquisition is now complete; the decision remains stop before a
new trial because evaluation prerequisites remain unverified. The text below
preserves the original proposal and initial audit.

Prepared September 9, 2026. **Proceed with data preparation and contract review;
stop before registering or evaluating a new campaign.** The observation screen
passes, but adjustment evidence and a separate untouched-evaluation plan are
unfinished. This proposal grants no registration or trading authority.

Machine-readable draft: `config/endpoint_data_proposal.json`.
Audit evidence: `artifacts/research_reset/endpoint_feasibility.json`.
Auditor: `ai_trading.tools.endpoint_feasibility`.

## What changes in the proposed data contract

For each fixed five-session opportunity, require only four exact observations:

| Observation | Required timestamp/value |
| --- | --- |
| Lookback close | Last regular-session minute's close, 61 sessions before entry |
| Previous close | Last regular-session minute's close immediately before entry |
| Entry | First regular-session minute's open at the fixed entry session |
| Exit | First regular-session minute's open five exchange sessions after entry |

The two closes are 60 exchange sessions apart. The previous close is complete
before the next session opens. Calendar/session helpers handle holidays, early
closes and daylight saving; missing observations never move the entry or exit.
An exit and the next entry can share a boundary, with exit-before-entry assumed.

All required bars must have unique timestamps and finite positive, coherent OHLC.
If one required observation fails for any ETF, exclude that entire common period
for every strategy/control. Keep DIA, IWM, QQQ, SPY, XLE and XLF. Do not interpolate,
forward-fill, substitute another timestamp or drop a symbol to improve results.
Interior minutes unused by the signal or execution convention are not mandatory.

Keep one equal-notional slot per symbol with inactive slots in cash: no leverage,
shorts, compounding or overlapping positions within a symbol. Cash and always-long
controls use exactly the same common opportunity mask, boundaries and capital
slots. This establishes alignment by construction, not a completed portfolio
replay or proof that opening bar prices were attainable.

## Feasibility result: observations are sufficient to consider a study

The audit used only the existing governed 2024–2025 SIP source. All six source
hashes matched their manifests. Prices were checked for validity at required
observations; no signal, selection, strategy return or baseline return was computed.

| Check | Observed result | Interpretation |
| --- | --- | --- |
| Fixed candidate periods | 88 | Same calendar grid as the prior slower study |
| Common valid periods | 88 | All required observations valid across all six ETFs |
| Periods required | 60 | Observation count can meet this requirement |
| Maximum possible selections | 528 | Upper bound only; actual selections unknown |
| Actual selections required | 100 | Remains untested without calculating the signal |
| Quarters represented | 7 | Each has 12–13 possible periods |
| Common masks and timing | Identical by construction | No baseline outcomes evaluated |
| Adjustment required by proposal | `all` | Existing source is `split`; requirement not met |

Quarterly possible periods: 2024Q2 13, 2024Q3 13, 2024Q4 13, 2025Q1 12,
2025Q2 12, 2025Q3 13, 2025Q4 12. Early 2024 supplies lookback observations.
Every individual symbol supports all 88 periods. The former whole-session rule
supported only four common periods because it required unused interior minutes.
This comparison concerns observation availability, not performance.

The feasibility result is an upper-bound screen. It does not establish sufficient
selected trades, statistical power, representativeness of other market regimes,
positive expected returns, or model superiority. Development data has already
been inspected in earlier research, so it cannot serve as untouched confirmation.

## Split/dividend treatment must be resolved before evaluation

Use a separately acquired and hashed provider `all`-adjusted research series and
verify the provider's adjustment conventions. Retain source vintage and corporate
action evidence. Check relevant splits/distributions against complete event records,
including symbol, event identity, effective/ex-date, amount/ratio and processing
date. Inspect the fields themselves; a provider completeness label is insufficient.

The current split-only series is useful for observation feasibility, but this
review does not certify dividend-consistent returns. Do not add cash dividends
again to an already dividend-adjusted research markout. Adjusted historical prices
remain research proxies; actual execution validation needs broker/quote prices,
fees and relevant cash flows. Do not claim a point-in-time corporate-action feed:
Alpaca explicitly describes possible publication/processing delays in its
[corporate-actions documentation](https://docs.alpaca.markets/us/reference/corporateactions-1).

The next data-preparation command is reviewable below. It has **not been run**;
it writes a separate dataset and cannot alter the consumed campaign:

```bash
./venv/bin/python -m ai_trading.tools.historical_training_backfill \
  --symbols DIA,IWM,QQQ,SPY,XLE,XLF \
  --start 2024-01-01 --end 2025-12-31 \
  --feed sip --adjustment all \
  --output-dir artifacts/endpoint_proposal/adjusted_data \
  --output-json artifacts/endpoint_proposal/acquisition.json
```

This requires the existing managed market-data credentials. Do not use
`--allow-incomplete` to force acceptance. Acquisition alone is not adjustment
verification. Once reviewed, update the draft's allowed audit-source adjustment
to match that new source and rerun the observation/validity checks before any
registration. Record the revised proposal hash.

## Cost-evidence path and research-only fallback

The available September 9 00:39 UTC paper-account snapshot contains 446 fill
activity rows with no fee amount and 81 fee activity rows with no fill/order
reference. The separately reconciled 400 local records remain without complete
per-fill totals. Those are different grains, not inconsistent execution counts.
This review reused the snapshot; it did not claim to fetch a newer account state.

The actionable evidence source is a broker-provided transaction confirmation or
equivalent record carrying account, execution identity, quantity, currency and
complete total charges, with an explicit completeness basis. Alpaca documents
trade-confirmation documents under its separate
[Broker API document endpoint](https://docs.alpaca.markets/us/reference/getdocsforaccount).
That endpoint's existence does not establish access through the configured paper
Trading API credentials. No Broker API access or document availability is assumed.

If such evidence remains unavailable, a separately approved historical study can
use **explicitly unverified assumptions**: retain 10 bps primary round-trip cost
and freeze adverse 20/40 bps scenarios before outcomes. These are stress assumptions,
not empirically calibrated costs or guaranteed upper bounds. Require positive
results under the adverse scenarios; do not choose a passing cost afterwards.
Apply full round-trip costs per completed position, including contiguous exit/
re-entry. Do not infer zero fees or allocate account charges to guessed executions.
Passing assumed costs cannot certify live execution economics.

## Proceed/stop decision

1. **Proceed:** review the proposed observation contract and prepare adjusted data
   plus corporate-action evidence. Timestamp/price-validity feasibility now supports
   this bounded preparation effort.
2. **Stop registration/evaluation:** adjustment treatment is unverified and a
   separately approved untouched-evaluation plan is missing. No campaign ID or
   trial budget has been created by this proposal.
3. **Before any later registration:** freeze the full hypothesis, baseline rules,
   costs, minimum sample requirements, statistical method, stopping criteria and
   evaluation plan. Recheck data feasibility on the exact approved input bytes.
   Registration requires separate authorization.

The original trial remains consumed and its report hash is unchanged. The existing
September 9–December 8 holdout is protected and was not read or repurposed. No
models, thresholds, symbols or returns were searched.

## Reproduction and checks

```bash
./venv/bin/python -m ai_trading.tools.endpoint_feasibility \
  --proposal config/endpoint_data_proposal.json \
  --acquisition artifacts/research_foundation/acquisition_sip_repaired.json \
  --output artifacts/research_reset/endpoint_feasibility.json
```

Regression tests cover exact endpoints, causal chronology, shared boundaries,
missing/duplicate/invalid required bars, valid price changes without selection,
source hash tampering, holdout rejection before price reads, and the distinction
between passed observation support and registration readiness. No changes were
made to scheduled research routing, model eligibility or trading decisions.

Validation completed: the changed-file validator with `--market-hours
--skip-runtime-smoke` passed lint, mypy, compilation and 123 selected tests,
including 10 new endpoint-feasibility regressions. A separate host health smoke
returned the existing HTTP 503 readiness gates. The proposal hash and both prior
campaign/report hashes were verified. `git diff --check` passed. No new study or
adjusted-data acquisition was run.
