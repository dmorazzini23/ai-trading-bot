# Requirements for resuming training

Prepared September 16, 2026. This is an operational checklist, not permission to
train, a registered new experiment, or a change to model qualification gates.
Authority: config/research_reset.json and docs/RESEARCH_RESET.md.

## Permission and development scope

- Scheduled training and new models remain paused. October 8 is a review date,
  not an automatic restart. Resumption requires an explicit review decision.
- Preserve the September 9–December 8 holdout. Do not use current paper losses
  to choose model parameters or evaluate alternatives on that holdout.
- Preserve consumed campaign budgets and inconclusive results. Any permitted
  future experiment needs its named hypothesis, development interval, data,
  fixed costs, stopping rule and evaluation criteria registered beforehand.

## Data acceptance before fitting

- Supply governed acquisition manifests with complete pagination, supported
  symbols and declared feed/adjustment. The ETF study does not establish data
  coverage for AAPL, AMZN or MSFT.
- Run the existing validated-data gate. Reject invalid OHLCV, duplicate/naive or
  off-grid timestamps, session/cadence violations and invalid feature values.
  Timestamp completeness alone is insufficient. Preserve explicit warmup and
  label-boundary exclusions rather than imputing them away.
- Declare and verify the live/training input contract: timeframe, finalization,
  source, feature definitions/order, history and label timing. Missing historical
  declarations remain unknown; changing current defaults does not repair them.

## Evidence required for qualification

- Establish account-scoped decision/order/fill links with explicit causal decision
  times, valid pre-decision quotes and complete session boundaries.
- Match broker cumulative quantities; require broker-sourced USD per-fill total
  fees for net-cost claims. No fee record does not mean zero fees.
- Compare simulation with observed execution only when order identity, side,
  symbol and arrival benchmark agree. Existing comparison support requires
  at least 30 paired orders across five sessions. This is measurement support,
  not profitability or permission to trade.
- Keep replay parity, freshness, provenance and promotion gates unchanged.
  Pass named candidate records through the existing registry evaluation;
  neither a successful training process nor a newer timestamp grants authority.
- Untouched evaluation must follow the existing holdout protocol when eligible;
  never substitute development performance or diagnostic paper fills.

## Current disposition

Training remains paused. September 15's ten fills reconcile by identity and
quantity but lack verified fees. Execution comparisons have no accepted pairs.
The latest selected model is stale and abstains. A fresh model would not resolve
these evidence gaps. Closing these gates is necessary evidence for the explicit
review, not a guarantee that any strategy will qualify.
