# Supported paper runtime path and configuration identity

**Status (2026-09-23):** this maps the packaged paper service and the
configuration identity now emitted by the next run manifest. It does not
approve a live profile or claim that all runtime-only settings are captured.

## Path to broker and accounting

1. `packaging/systemd/ai-trading.service` copies `.env` to the protected
   runtime environment file, applies Alembic migrations, then starts
   `python -m ai_trading`. The main API and health route share port 9001.
2. `ai_trading.core.run_all_trades_worker` obtains the validated
   `TradingConfig` and policy through its prelude. The prelude writes
   `runtime/run_manifest.json` on the first successful cycle preparation.
   Market data and strategy decisions flow through `ai_trading.core.bot_engine`
   and `ai_trading.core.netting_symbol_cycle`; decision events are recorded by
   `ai_trading.oms.decision_events` with their causal timestamp basis.
3. `ai_trading.core.netting_submit_execution` hands the order contract to
   `ai_trading.execution.live_trading.ExecutionEngine.execute_order`.
   Pretrade, launch-profile, model/replay, quote, provider, broker-freshness and
   OMS checks govern openings. `ai_trading.execution.engine.OrderManager` and
   `ai_trading.oms.intent_store.IntentStore` persist submission identity before
   the Alpaca SDK call. An unresolved `SUBMITTING` intent blocks subsequent
   openings until broker identity reconciliation resolves it.
4. Execution capture records order/fill events and TCA when the source supplies
   the required evidence. `ai_trading.tools.execution_evidence_reconciliation`
   checks decision/order/fill/TCA joins; `broker_accounting_evidence` checks
   broker execution quantities. Verified fee totals require separate broker
   evidence. The unsupported pre-September 7 legacy interval remains excluded;
   see `ACCOUNTING_AND_RESEARCH_DECISION_20260923.md`.

The canonical engine's market, limit, replacement and cover submissions now
check the live OMS owner immediately before reaching the Alpaca SDK. The
standalone `alpaca_api.submit_order`, `bot_engine.safe_submit_order`, and
`ExecutionEngine.safe_submit_order` helpers reject live mode; they remain
available for their supported paper/test use. A live opposite-side cover now
requires fresh broker positions and open orders, an account identity, and a
durably claimed OMS intent before submission. An ambiguous broker response
leaves that intent unresolved for identity-based reconciliation. The lower-level
`ExecutionEngine._submit_order_to_alpaca` now requires a `SUBMITTING` intent
from the canonical execution route, with matching client identity, symbol,
side and authorized quantity. A lost live submit response is not blindly
retried or moved back to a reusable intent state. Live cancel-and-resubmit
limit replacements are held before the helper cancels anything until a durable
replacement pretrade and identity contract is implemented. Pending-order
maintenance may still cancel a stale opening order under its timeout policy.
It preserves a broker-open closing order when the broker supplies a matching
closing `position_intent` or a matching durable OMS intent verifies the closing
purpose, even if that policy would otherwise cancel it.
This limitation must be resolved or explicitly accepted before live
activation. Alpaca's [replace-order contract](https://docs.alpaca.markets/us/reference/replaceorderforaccount)
does not permit native replacement of `pending_new` orders and warns that a
successful replacement response does not guarantee the old order was replaced.
The durable identity check is one
boundary and does not independently prove the upstream model, risk and quote
checks; those remain in `execute_order`.
The existence of these helpers is not evidence of current runtime use. The current
paper service reports `paper_trade`, diagnostic-only operation and a blocked
qualified-model readiness gate; this is a September 23 observation, not a
permanent configuration guarantee.

## Configuration contract

The packaged service uses `.env` as its source and sources its synchronized
runtime copy before Python starts. The resulting process environment is parsed
through `ai_trading.config.management`. `TradingConfig.from_env` validates
declarative bounds and choices, resolves aliases, and records the effective
trading-mode source and precedence policy. `resolve_launch_profile` separately
applies profile defaults and environment overrides; live-profile numeric
overrides may tighten the defaults but cannot loosen them. Both resolved
surfaces must be checked for a deployment.

| Setting | Unit and meaning |
| --- | --- |
| `max_gross_exposure`, `max_symbol_exposure` | Fraction of account equity |
| `max_daily_loss`, `max_notional_per_order` | USD |
| `max_order_count` | Opening attempt count per profile budget |
| `max_quote_age_ms` | Milliseconds |
| `max_spread_bps` | Basis points |
| `order_timeout_seconds` | Seconds |

`config_snapshot_hash` now includes a digest of all declarative `CONFIG_SPECS`
values, with masked credentials represented only as configured/absent. The
startup manifest's `resolved_config_hash` uses the same canonical algorithm as
decision/order lineage. It also records the separate resolved launch-profile
payload and `launch_profile_hash`, because profile-specific overrides are read
outside `TradingConfig`. The raw secret values are neither written to the
manifest nor included in the digest. Any change to a declared setting or
profile override produces a different corresponding identity.

This is a version boundary: an older run manifest used a different JSON hash
format and did not capture all declared settings or the launch profile.
Historical hashes must not be retroactively rewritten or compared as though
they had the new scope. Runtime `get_env` settings outside `CONFIG_SPECS` and
the launch-profile resolver, the applied database revision, the exact loaded
model artifact and checkout dirty state are still separate evidence gaps.
Before a live deployment, a release record must bind tested commit, effective
configuration and profile hashes, applied schema revision, model artifact
identity and broker exposure, with a migration-aware rollback rehearsal.
