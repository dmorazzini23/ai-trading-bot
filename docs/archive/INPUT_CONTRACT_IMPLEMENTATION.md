# Input contract implementation and evidence review

September 13, 2026. Calendar filtering is corrected; finalized batches and
inference inputs have versioned provenance. Existing qualification gates remain
unchanged. Full parity and real execution are not established by these changes.

## Calendar and batch evidence

Intraday regular-session normalization now uses the canonical exchange calendar,
excluding holidays and rows at/after early closes. Daily and explicitly extended
session paths retain their existing semantics. Five-minute finalization remains
close-boundary plus bounded grace.

`DAY_SLEEVE_FINALIZED_BATCH` logs requested history bounds, actual first/last
bar starts, retained count, session policy, grace and input hash. This runs even
when a stale model prevents inference. `DAY_SLEEVE_INFERENCE_INPUT` adds feature
column order, feature-row hash and training-contract comparison before scoring;
the provenance is also included in decision debug data. Neither log submits orders.

Provider frame attributes retain requested feed/adjustment separately from source
feed/provider and effective adjustment. Direct Alpaca response paths label
adjustment evidence as provider_request: this records the request that produced
the response, not an independent corporate-action audit. Fallback paths without
such evidence remain unknown. Hashes cover ordered index/columns/values using
pandas split JSON, ISO nanosecond timestamps and 15-digit numeric serialization;
they are diagnostic identities, not cryptographic certification of broker truth.

## Versioned training/serving contract

`day_input_v1` requires explicit feed, adjustment, timeframe, session policy,
history policy, finality policy and feature version. The model loader carries
declared input_contract metadata if present. Missing historical metadata is never
backfilled from current settings. `compare_input_contracts` returns matched only
when both versions and all required declarations match; unknown or differing
fields yield unverified with reasons. Matching has no qualification authority.

Example declaration shape (values must come from verified acquisition/build
evidence, not this example):

```json
{
  "version": "day_input_v1",
  "feed": "<verified feed>",
  "adjustment": "<verified adjustment>",
  "timeframe": "5Min",
  "session_policy": "canonical_exchange_regular_session_v1",
  "history_policy": {"kind": "rolling_calendar_days", "days": 10},
  "finality_policy": {"bar_label": "start", "grace_seconds": 2.0},
  "feature_version": "<declared feature version>"
}
```

This schema does not authorize training or prescribe a new model configuration.
Prior models without the declaration remain unverified for input parity. Existing
freshness, lineage and qualification checks still decide serving eligibility.

## Evidence review and configuration decision

Keep configuration unchanged. Current live request settings are IEX/all with a
10-calendar-day default history. The selected model records 60 training days but
lacks a complete historical input contract. Current reference-training code uses
delayed_sip/raw; the development diagnostic used SIP/split. None establishes the
selected model's original feed/adjustment. Nine identical-history feature checks
passed, but truncation changes recursive indicators; numerical effects were not
translated into prediction or performance claims.

Required next observation: during a normal scheduled session, inspect finalized
batch evidence and, only if inference is otherwise eligible, its feature-input
record. Check actual history, provider/fallback and missing adjustment fields.
Do not force trades or bypass stale-model restrictions to produce this evidence.
This future-session observation cannot be completed during Sunday's market closure.

## Validation, risk and rollback

Tests cover holidays/early closes, daily/extended preservation, unknown fallback
adjustments, hash changes, missing/mismatched contracts, no-data batches and
day-sleeve serving integration. See CODEX_HANDOFF.md for final checks/deployment.
Primary behavior change is removing out-of-session intraday rows, which can alter
indicator inputs on holiday/early-close windows. Provenance adds structured log
volume and hashing work proportional to the requested batch size. Revert these
calendar/provenance edits to roll back; preserve older replay and research work.
No model, fee assumptions, training budget or holdout policy changed.
