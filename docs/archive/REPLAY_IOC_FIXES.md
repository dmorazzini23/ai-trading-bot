# Replay IOC handling — September 12, 2026

IOC orders receive one fill attempt against the market observation at submission
time. Any remainder is canceled immediately, including partial fills, missing
quotes, unmarketable limits and probability-based non-fills. Later processing
cannot fill them. Immediate fills update exposure before the next order cap check.
Unsupported time-in-force values such as FOK still fail explicitly.

This follows [Alpaca IOC semantics](https://docs.alpaca.markets/us/docs/orders-at-alpaca).
Prices, liquidity, spread, fees and random fill outcomes remain simulation
assumptions, not broker execution evidence. Saved assumptions identify the IOC
price basis; summaries persist IOC remainder cancellations with effective and
observed timestamps. No cost, qualification, model or research gates changed.

## Verification

- Focused broker and governance regressions: 51 passed; full, partial, missing
  quote, unmarketable, delayed-processing and exposure-cap cases covered.
- Non-sending incident snapshot passed.
- Isolated production-settings replay wrote
  `/tmp/replay-ioc-check/output/replay_hash_20260912.json`, SHA256
  `92c8a585600b184668e07f5d20e0eb7d8b4af8d148f430bcebd5e101ad70ca9b`.
- Candidate: 83 fill events, 77 usable markouts, 6 horizon exclusions,
  60 expiry events; net edge -9.750137 bps.
- Baseline: 539 fill events, 525 usable markouts, 7 horizon and 7 missing
  subsequent-price exclusions, 486 expiry events; net edge -7.181490 bps.
- Usable plus excluded counts reconcile in both summaries. All observed bounded
  fills precede expiry. Three baseline IOC markouts have zero fill delay.
  No IOC cancellations occurred in this dataset; regression tests cover them.
- Replay correctly remains blocked on positive edge, minimum sample count and
  edge non-regression. This is operational diagnostic evidence only; holdout and
  research-reset restrictions remain intact. Production artifacts were not replaced.

See CODEX_HANDOFF.md for final changed-file validation and service smoke results.
Rollback should revert only these IOC edits in simulated_broker.py, event_loop.py,
bot_engine.py and their regression tests, preserving earlier contract/expiry work.
Main residual risk is sparse replay observations and modeled liquidity; these
results do not establish tradable profitability.
