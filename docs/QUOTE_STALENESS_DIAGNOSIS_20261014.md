# October 14 decision-quote staleness diagnosis

## WORKLOG

Named evidence gap: the Oct 13 research reset scorecard rejected 1,896 of
2,399 decision quote rows as `quote_age_over_1000ms`. 1,629 were
`reported_fresh_but_record_stale` and 23 were `quote_observation_time_invalid`.
All 22 fills had `causal_quote_unverified`. This work used only code,
`decision_records.jsonl` (the same 2,399-row window) and the journal. No gates,
thresholds, models, the 1,000 ms requirement or the runtime checkout changed.

## Where the time goes

Window rows, seconds (n=2,398; p10 / p50 / p90 / p99):

| Interval | p10 | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| quote exchange ts → observed (`quote_observed_at`) | 0.03 | 0.27 | 2.43 | 14.1 |
| observed → `recorded_at` | 5.78 | 7.87 | 25.8 | 155 |

The quote is usually fresh when observed (1,606 of 1,873 valid observations
were ≤1 s old). The time is lost after that, inside the cycle. The 19:50 UTC
Oct 13 cycle (37 s) is typical:

- 17.9 s: the cycle `now` is captured by governance.
- 19.1–21.3 s: the prerank ML-shadow loop (`_prefetch_ml_shadow_quote`) fetches
  one quote per symbol into the shared `runtime_state` per-symbol cache.
- 23.5 s: day-sleeve finalization. 27.5 s: policy gates.
- 27.5–30.8 s: per-symbol netting. Decision records are built here.
  `DecisionRecorder` copies the *cached* prerank quote.

The clock is NTP-synchronised UTC, so this is not clock skew.

## Root causes

1. **Recording bug: the decision time was the cycle start.** The netting
   submit prelude passed the cycle-level `now` as `decision_timestamp`. But the
   arrival quote that the order decision actually uses is fetched later, per
   symbol. 487 of 515 explicit-time rows (464 `submit_runtime`, 23 `submitted`)
   therefore had a quote timestamp *after* the decision time. This fails the
   causal check `0 ≤ decision − quote ≤ 1 s` for every fill.
2. **Recording bug: the record got the wrong quote.** The decision record took
   its quote from the shared runtime snapshot (usually the prerank prefetch)
   rather than the submit-arrival quote. For `submitted` rows it combined the
   arrival `quote_timestamp` from the submit metrics with the snapshot's older
   `quote_observed_at`, bid/ask and age. That accounts for all 23
   `quote_observation_time_invalid` rows: observation 6–24 s *before* the quote.
   Example: AMZN 19:50, record quote 20.17 s (prefetch), but the
   `STALE_QUOTE_BLOCK` gate evaluated a 25.14 s arrival quote.
3. **Real pipeline delay, not a recording bug:** 1,669 `decision_filter`, 158
   `execution_approval` and most `precheck` rows have no per-symbol quote at
   decision time. They reuse the prerank quote 4–9 s later (median 7.9 s). These
   rows never reach submission. The scorecard is right to reject them.

## Patch

- `netting_submit_prelude`: the explicit decision time is the wall clock right
  after the arrival quote is resolved. The prelude returns
  `decision_quote_evidence` (arrival quote ts, bid/ask, source, observation time,
  age at observation).
- `netting_symbol_cycle`: submit and precheck decision records carry that
  evidence, not the runtime snapshot.
- `decision_log`: snapshot prices, age and observation time are never combined
  with a different explicitly recorded quote. A missing field stays missing.

Unchanged: `pretrade_intent.quote_age_ms` is still computed against the cycle
`now`. This preserves the existing gate input (see follow-up). The cycle `now`,
client order IDs, thresholds and the 1,000 ms audit rule are also unchanged.

## Expected effect

After deployment, a filled order's decision row pairs the explicit decision time
with the quote the order used. `causal_quote_unverified` should then clear
whenever the arrival quote was ≤1 s old at fetch. The prefetch distribution
(~86% ≤1 s) suggests that is most fills. Submit's existing 2,000 ms
`STALE_QUOTE_BLOCK` bounds the rest. Complete chains will **still be 0** while
`verified_total_fees_missing` affects all fills. The research spread audit
measures against `recorded_at`, so `submitted` rows recorded after the order
lifecycle (often minutes) remain rejected there. That is by design.

## 893f491 (symbol-scoped broker reads)

That commit targets account-wide order/history scans in submit, reconciliation
and EOD paths. It is expected to shorten slow-cycle tails (p99 ≈155 s
observed → record; the 110 s critical cycles) and the delay before `submitted`
records are written. It does **not** touch the prerank-to-netting interval:
day-sleeve and policy compute dominate there, with a median of 7.9 s. So it will
not by itself bring `decision_filter` rows under 1,000 ms. Only the recording
fix above clears the fill causal-quote gap.

## Follow-ups (not done)

- `PretradeOrderIntent.quote_age_ms` is clamped to 0 when the arrival quote
  postdates the cycle start, so `EXECUTION_QUALITY_STALE_QUOTE_BLOCK` never sees
  a real age. The separate submit `STALE_QUOTE_BLOCK` (2,000 ms) still works.
  Correcting this changes a trading gate input and needs owner approval.
- `bot_engine` quote-gate `_publish_quote_state` updates bid/ask without
  `quote_timestamp`, which can leave a mixed per-symbol snapshot. The recorder
  guard limits the impact. The cache itself is unchanged because live gates read
  it.

## Validation

See the session report / commit message for commands and results.
