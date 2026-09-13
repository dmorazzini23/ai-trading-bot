# Live versus training bar contract — September 13, 2026

Conclusion: identical-history formula checks passed previously, but live input
equivalence is not established. This audit changes no runtime settings or models.

| Dimension | Current live path | Training / development evidence |
| --- | --- | --- |
| History request | 10 calendar days by default, bounded 7-60; runtime env has no override | Selected model metadata records 60 training days; recursive-indicator initialization equivalence unverified |
| Timeframe | 5Min, start-labelled | Selected model declares 5Min and one-bar labels; elapsed labels remain irregular |
| Finalization | Keep bar only when start + five minutes <= now minus two seconds | Development uses complete exact minute groups; historical acquisition does not prove live arrival latency |
| Feed | Execution role, effective priority alpaca_iex then yahoo | Current training reference role resolves delayed_sip; governed development audit uses sip |
| Adjustment | Effective settings: all | Current reference get_bars defaults raw; development acquisition is split |
| Session filter | Weekday 09:30-16:00 New York in normalize_bars | Development audit uses exchange calendar including holidays and early closes |

Selected model metadata does not establish the original training feed/adjustment.
Current training code is not proof of those historical values. Do not relabel
the existing artifact or assume the new development bars reproduce its training.

## Evidence and limitations

Effective values were loaded through configuration management from the packaged
runtime environment. `_netting_sleeve_fetch_start` uses the day-sleeve history
setting; `_run`'s netting path calls `get_bars_batch` with default execution role.
`get_bars` resolves execution priority and configured adjustment; the reference
branch instead uses `get_reference_feed` and explicit-or-raw adjustment.
Settings evidence: /tmp/live-contract-settings.log. No secrets were printed.

The service journal since reboot (2,628 lines inspected, saved privately under
/tmp/live-contract-service.log) contains no matching DAY_SLEEVE/BAR_FETCH/5Min/
FETCH_BARS entries. Consequently requested settings are verified, but actual
returned feed, first/last timestamps, count, fallback use and finalized-batch hash
cannot be certified from this weekend log. No current holdout bars were fetched
for model evaluation, no predictions/returns computed, and no new trial claimed.

19 existing day-sleeve serving tests passed. Additional synthetic assertions
verified exclusion at bar-close+1 second and inclusion at bar-close+2 seconds.
Synthetic July 4 and July 3 post-early-close rows are retained by normalize_bars:
the calendar discrepancy is demonstrated, not merely inferred. Provider behavior
may supply no such rows, but the normalizer itself does not enforce the exchange
session. Evidence: /tmp/live-contract-boundaries.log.

Health: broker fresh, zero positions/orders, existing required_model_stale and
replay_live_parity_gate_failed flags. No restart or deployment was required.

## Next corrective scope

1. Fix regular-session normalization to use the canonical exchange calendar,
   with holiday/early-close regression tests; preserve bar finalization.
2. Persist actual inference-batch provenance (requested/effective feed and
   adjustment, history bounds, finalized count, source and feature-input hashes).
   Verify it on an ordinary scheduled session without forcing trades.
3. Require a versioned matching training/serving history and adjustment contract
   before claiming full parity. Do not guess missing historical metadata or switch
   live feeds/adjustments to improve results. Any new model remains separately
   governed and training stays paused.

Full execution and attributable fee evidence remain separate unresolved gaps.
