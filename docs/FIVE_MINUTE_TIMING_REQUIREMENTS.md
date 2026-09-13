# Five-minute timing requirements and development coverage

September 13, 2026. This is a proposed measurement contract and timestamp-only
feasibility audit. It does not change the selected model, runtime execution,
qualification metrics, campaign budget, or the registered slower ETF hypothesis.

## Timing requirements

- Decision time D is a regular-session five-minute boundary, aligned to session
  open. Inputs use the five completed minute bars with start timestamps D-5
  through D-1. The last bar becomes complete at D; no partial D bar is an input.
- Entry reference is the next minute boundary, D+1 minute. The one-minute delay
  is an explicit analysis convention, not measured latency or guaranteed fill.
- Exit reference is D+6 minutes, exactly five minutes after the entry reference.
  Thus decision-to-exit is six minutes; this is not a decision-plus-five-minute
  label. Execution must preserve its actual timestamps and cannot assume a bar
  open was fillable. Unfilled/late entries cannot silently shift the exit.
- Require every minute timestamp D-5 through D+6 inclusive, uniquely observed
  and within the same regular session. Exit must precede session close; respect
  canonical exchange holidays and early closes. Never carry, interpolate or
  replace a missing timestamp with the next available observation.
- This audit describes candidate slots on a five-minute grid. Slots overlap in
  input windows and are not independent trades. Signal selection, actual fills,
  quote freshness, feature lookbacks beyond five minutes and portfolio constraints
  can reduce usable opportunities further.

These conventions must be explicitly accepted in a separately governed strategy
evaluation before any return calculation. They are not retroactive semantics for
the existing model. Its metadata says one 5Min bar, but label durations range from
five minutes to about 89.67 hours. Those labels do not meet this elapsed-time
contract; no labels were rewritten and no model was fitted during this audit.

## Data boundary and reproducibility

Use only the existing governed SIP, split-adjusted, 1Min ETF acquisition for
2024-01-01 through 2025-12-31. The canonical provenance audit verifies identity,
quality, symbol paths, hashes, timezone presence and source stability before use.
The new audit rechecks hashes around timestamp-only parsing and rejects actual
timestamps outside development. No 2026 price data or protected holdout is used.

Symbols: DIA, IWM, QQQ, SPY, XLE, XLF. The current AAPL/AMZN/MSFT universe is absent;
ETF coverage cannot establish selected-model coverage or applicability. This is
previously used development data, not untouched evidence.

Run from the repository root:

```bash
./venv/bin/python -m ai_trading.tools.five_minute_coverage
```

Output: `artifacts/research_reset/five_minute_coverage.json`, containing per-session
and per-symbol counts, source hashes and unchanged campaign-ledger hashes. Hash
verification reads source bytes; only timestamps enter the analysis. No prices,
signals, returns, fitted models, order submissions or trial claims are computed.
Exclusion categories are disjoint: session boundary first, duplicate timestamps
second, missing timestamps third; otherwise complete timestamp coverage.

Regression tests cover missing input/exit timestamps, duplicate timestamps,
session boundaries and refusal to substitute a later observation. No runtime
service deployment is needed for this standalone audit.

## Observed coverage

502 sessions per ETF, including canonical early closes; 3,012 symbol-session
records. Each symbol has 38,438 candidate slots, of which 502 are excluded at
session boundaries. There are no duplicate-timestamp slot exclusions.

| Symbol | Complete timestamp windows | Missing-timestamp windows |
| --- | ---: | ---: |
| DIA | 36,088 | 1,848 |
| IWM | 37,936 | 0 |
| QQQ | 37,936 | 0 |
| SPY | 37,936 | 0 |
| XLE | 37,936 | 0 |
| XLF | 37,936 | 0 |

The audit confirms timestamp support for this proposed window in five ETFs,
with explicit gaps in DIA. It does not certify model-ready observations: prices,
features, actual signal opportunities and executable quotes were not assessed.
Current stock-universe coverage remains unavailable. Campaign-ledger hashes were
unchanged. No additional trial is authorized by these counts.
