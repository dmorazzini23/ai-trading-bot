# SIP entitlement and model inputs — September 17, 2026

Read-only checks at 03:35 UTC; no orders, training, feed changes or purchases.

## Entitlement

Current credentials rejected both SIP latest quotes (AAPL/MSFT) and recent
five-minute-window minute bars (AAPL): `subscription does not permit querying
recent SIP data`. Evidence: `/tmp/sip-access-review.json`. Earlier historical
SIP requests succeeded (see SKEW_EVIDENCE_AND_FEED_COMPARISON_20260917.md).
Historical access therefore does not establish real-time access. A live SIP
migration is blocked with these credentials; any subscription purchase requires
an explicit user decision.

Current effective declarations: market/execution feed IEX, reference delayed_SIP,
live adjustment all, historical-backfill feed IEX and adjustment raw. These are
configuration declarations, not proof of original model training inputs.

## Selected day model declarations

Inspected registry metadata for
`ml_edge-histgb-236ba0fe-20260717200350-4045d5c5` under
`/var/lib/ai-trading-bot/models/models/` and its referenced artifact manifest
`/var/lib/ai-trading-bot/models/after_hours/ml_edge_histgb_20260717_200036.joblib.manifest.json`.
Registry manifest metadata declares IEX, 5Min, feature contract
`day_sleeve_ml_feature_contract_v1`, and lookback_days=60. It does not declare
the full versioned input contract: adjustment, session, history semantics and
finality are missing. A 60-day lookback alone does not prove rolling-history
semantics. The artifact manifest also lacks an input_contract.

The generic configured trained_model.pkl manifest was inspected separately;
it likewise lacks an input contract and is not substituted for the selected
day model's evidence. Existing compare_input_contracts correctly requires all
seven fields and the contract version; compatibility remains unverified.
Do not populate old training declarations from today's environment or infer
that acquiring SIP makes the existing model compatible with SIP.

## Remaining work

- Decide whether real-time SIP access is worth acquiring before planning a
  feed migration; no subscription or feed change was made.
- Recover original training provenance if available. If it cannot be recovered,
  leave compatibility unverified until an explicitly approved, governed training
  run produces the complete contract. Preserve research reset and holdout rules.
- Inspect the next natural skew event for the newly deployed diagnostics.
- Verify closeout behavior when a natural session has remaining exposure.
  Do not manufacture positions or warnings to obtain operational evidence.

This was a read-only external/metadata audit plus documentation. No runtime
code changed, so no regression tests or service restart were needed.
