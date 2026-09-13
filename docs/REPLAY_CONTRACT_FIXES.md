# Replay order and observation contract fixes

September 12, 2026. Implements the corrective scope from
docs/REPLAY_EXECUTION_DRAG_AUDIT.md without changing qualification thresholds.

- The recorder and journal keep submission_status separate from order_type.
  No-intent observations are not_submitted; that value is no longer emitted as
  an executable order type. Missing order intent is explicitly incomplete metadata.
- Opportunity replay requires explicit opportunity_order_type or order_type. Unsupported or
  missing types produce unsupported_or_missing_order_type diagnostics; historical
  intent is not inferred. Market and limit are the simulator's supported types.
- SimulatedBroker rejects unsupported types before creating or scheduling an
  order. Legacy input using not_submitted as type now fails instead of simulating
  an unconstrained order. Stop variants remain unsupported rather than approximated.
- Submission status survives replay normalization and order construction.
- Saved markouts include decision/reference, fill and markout timestamps, actual
  decision-to-fill delay and markout interval. In this replay, decision/reference
  time is the simulated order submission timestamp; it is not a broker timestamp.
- Fill events and saved markouts include the model identifier, seed, fill/partial
  probabilities, spread/volatility/fee assumptions, jitter range, price used,
  scheduled fill time and whether limit constraints apply. These are explicitly
  labeled simulator assumptions, not observed execution costs.

The next-observation metric, 24-hour maximum, profitability/freshness/provenance
gates, research reset and selected model remain unchanged. Existing artifacts are
not rewritten. New evidence can contain fewer rows when historical order intent is
unknown; this is a reported evidence limitation, not permission to weaken gates.

Regression coverage checks rejected types create no order/fill, JSON-serialized
timing and assumptions, explicit opportunity type with separate submission status,
missing-intent rejection and recorder/journal propagation. Synthetic tests exercise
replay behavior without reading the protected holdout or submitting broker orders.
Final validation and runtime results are in docs/CODEX_HANDOFF.md.

Rollback only these scoped code/test changes if needed, preserving prior fixes
and all settings. Restoring old code also restores unsafe unsupported-type
semantics; do not treat old replay reports as newly validated evidence.
