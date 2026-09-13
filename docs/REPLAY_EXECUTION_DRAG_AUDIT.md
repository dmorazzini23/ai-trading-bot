# Execution-price drag audit — September 11, 2026

Conclusion: the arithmetic reconciles, but execution semantics and missing timing
evidence prevent interpreting the 14.339406 bps drag as validated execution cost.
This is a bounded code and saved-artifact audit, not a strategy experiment.

Artifact: /var/lib/ai-trading-bot/runtime/replay_outputs/replay_hash_20260911.json.
Its hash and gross-minus-drag reconciliation are recorded in
docs/ROLLBACK_REPLAY_MODEL_DECISION.md. No source dataset was replayed or modified.

## Findings

1. Unsupported order semantics: 2,512 of 3,174 candidate markouts carry
   order_type=not_submitted; the other 662 carry limit. The governance strategy in
   core/bot_engine.py copies order_type into the proposal, event_loop.py passes it
   into SimulatedBroker, and the broker applies limit-price constraints only when
   type equals limit. Thus not_submitted is effectively handled as a market-like
   order even though it describes submission state, not executable order intent.
   A synthetic buy with limit_price=100 and observed market price=110 produces no
   fill for type=limit but one fill for type=not_submitted. This reproduction used
   invented data only. The historical intended order type is not established.

2. Assumed price impact: event_loop.py calls submit_order without spread_bps or
   volatility_pct. Defaults are 8 bps and 0.01. The simulator adds/subtracts half
   the spread, a uniform volatility component between 0 and 35 bps, and jitter
   between -8 and +8 bps, subject to limit clamping. These are simulator assumptions,
   not measured per-order spread or impact. Current code's unconstrained expected
   adverse adjustment is about 21.5 bps before reference-to-fill movement. This
   expectation does not directly predict the saved 14.34 bps mean.

3. Variable horizon: _replay_summary_metrics selects the first same-symbol price
   timestamp strictly after the fill via bisect_right, discarding observations
   more than 24 hours away. The 24-hour setting is a ceiling, not a fixed holding
   period. Arrival/reference-to-fill movement and the return denominator enter
   execution_drag_bps, which is defined as gross_edge_bps minus net_edge_bps.
   This is not linked entry-to-exit trade accounting.

4. Missing audit evidence: saved markout rows contain prices and client order IDs,
   but no decision, reference, fill or markout timestamp. The artifact also omits
   the raw order/event arrays. Actual delay/horizon distributions cannot be
   reconstructed reliably from this artifact alone. Matching the intended strategy
   horizon is therefore unverified, rather than confirmed or disproven.

## Numerical checks

All 3,174 saved candidate markouts have zero fee_amount. There are 586 negative
execution-drag observations; range -330.381560 to 396.963416 bps. Such values are
consistent with reference/fill movement being included and are not necessarily
arithmetic errors. Mean gross -3.941979 minus drag 14.339406 equals net -18.281386.
Neither the negative gross proxy nor its subgroups establish live strategy returns.

## Next corrective scope

- Separate submission state from intended execution order type; reject unsupported
  types instead of silently treating them as market-like. Do not guess historical
  intent or rewrite old artifacts.
- Persist decision/reference, fill and markout timestamps and measured intervals;
  record fill assumptions explicitly alongside each observation.
- Declare the metric's next-observation horizon and keep it distinct from the
  strategy's intended holding period. Validate any new fixed-horizon metric under
  the authorized research protocol rather than changing the gate until it passes.
- Use governed execution evidence to validate price-impact assumptions. Missing
  evidence remains a qualification blocker; do not zero costs to manufacture edge.

The existing report overlaps the protected holdout. No holdout-derived selection,
tuning, new backtest, live order, model replacement or configuration change occurred.
Validation: synthetic unsupported-order reproduction passed; docs-only validator
and git diff --check run. Runtime checks are unnecessary for this documentation-only
audit. The existing negative replay gate and model abstention remain appropriate.
