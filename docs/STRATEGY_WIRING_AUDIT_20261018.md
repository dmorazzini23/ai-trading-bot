# Strategy wiring audit: October 18, 2026

**Question:** `STRATEGIES` in `.env` names six strategy classes. Do any of them
reach the live decision and order path, or are they dead code in the current
configuration?

**Answer:** none of them run. `get_strategies()` instantiates all six on the
first cycle each session, and nothing calls them after that. The live runtime
uses the netting pipeline (`AI_TRADING_NETTING_ENABLED=1`, also the default),
and that branch returns before `run_multi_strategy()`, the only caller of
`generate()` / `generate_signals()`. The netting pipeline gets its scores from
`_score_from_bars()` plus the day-sleeve ML model. It knows only the sleeve names
`day`, `swing` and `longshort`, and has no way to take a `StrategySignal`.
`allocation_weights == {"day": 1.0}` comes from normalizing the weights over the
one sleeve left after the sleeve whitelist. It does not come from the March
weights.

The scope was read-only. No code or config was changed. Line numbers refer to
`1e7c849f4`, which is the current `HEAD` of the runtime checkout
(`/home/aiuser/ai-trading-bot`). Between `ad499f4fe` and `1e7c849f4`, the only
`bot_engine.py` hunks fall in rollout-governance code (around lines 41600–41690).
Lines above that are identical. Lines below it are shifted by up to +25 relative
to `ad499f4fe`. `strategy_cycle.py`, `run_all_trades_*.py`,
`netting_cycle_setup.py`, `horizons.py` and `ai_trading/strategies/` are
unchanged between the two commits.

## 1. Call graph: loader to order

```
run_all_trades_worker            core/run_all_trades_worker.py:201  passes be.get_strategies
 └─ prelude                      core/run_all_trades_prelude.py:356-358
      runtime.strategies = get_strategies()      (once per session; _strategies_loaded)
       └─ get_strategies()       core/bot_engine.py:19111
            reads STRATEGIES     bot_engine.py:19142  (after settings STRATEGIES_WANTED)
            REGISTRY lookup      strategies/__init__.py REGISTRY → cls() per name
            logs "Loaded strategies: …"  (logger_once)
 └─ execute_run_all_trades_cycle core/run_all_trades_execution.py:189
      if _netting_pipeline_enabled(runtime):     run_all_trades_execution.py:305
          _run_netting_cycle(...)                 :322   ← LIVE PATH
          return                                  :396   ← strategies never touched
      ... legacy path ...
      _process_symbols(...)                       :623   (SignalManager per-symbol path)
      be.run_multi_strategy(runtime)              :779   ← ONLY consumer of the six
        └─ run_multi_strategy_cycle  core/strategy_cycle.py:12
             for strat in ctx.strategies: strat.generate(ctx) | generate_signals(...)   :57-75
             (optional RL signals, signal enhancement / holding logic)
             ctx.allocator.allocate(signals_by_strategy)  StrategyAllocator           :312
             _resolve_order_intent(...)                                               :637
             ctx.execution_engine.execute_order(...)                                  :685
```

Live netting path, for contrast:

```
_run_netting_cycle                         bot_engine.py:42066
 └─ prepare_netting_cycle_inputs           bot_engine.py:42349 → core/netting_cycle_setup.py:277-324
      sleeves = build_sleeve_configs(cfg) filtered by enabled          :277
        (core/horizons.py:63-76: names must be day|swing|longshort, else ValueError)
      AI_TRADING_RUNTIME_SLEEVE_WHITELIST filter, logs RUNTIME_SLEEVE_WHITELIST_APPLIED  :278-289
      _maybe_update_allocation_state(...)                                :318
      allocation_weights = _allocation_weights_for_sleeves(cfg, sleeves) :324
 └─ per (sleeve, symbol):
      score, confidence = _score_from_bars(df)          bot_engine.py:42685  (fast/slow return z-score, :35489)
      _score_day_sleeve_with_ml(...) blend              bot_engine.py:42694  (:7446)
      proposal = compute_sleeve_proposal(sleeve, ...)   bot_engine.py:42721
      proposal.target_dollars *= allocation_weights[sleeve.name]   :42794
 └─ netting / rank / approval / submit     core/netting_symbol_cycle.py, netting_rank_runtime.py,
                                           netting_submit_execution.py (no reference to
                                           .strategies, StrategySignal or REGISTRY)
```

A repository grep finds no consumer of `StrategySignal` or `generate_signals`
outside `ai_trading/strategies/`, `strategy_allocator.py`, `strategy_cycle.py`,
`bot_engine.py`, `risk/engine.py`, `signals/`, `core/interfaces.py` and
`rl_trading/`. None of these sits on the netting path. No research tool,
replay or backtest imports the six classes. Only
`tests/strategies/test_sleeves_phase2_local.py` exercises them.

## 2. How `allocation_weights` becomes `{"day": 1.0}`

`_allocation_weights_for_sleeves` (bot_engine.py:38035-38053):

1. Because `AI_TRADING_ALLOCATION_ENABLED=1`, it loads the persisted weights from
   `AI_TRADING_ALLOCATION_OUTPUT_PATH=runtime/allocation_state.json`, resolved
   relative to the service `WorkingDirectory=/home/aiuser/ai-trading-bot`. That
   file holds the March 10 weights `{day: 0.4, longshort: 0.25, swing: 0.35}`.
2. It keeps only the sleeves passed in. `build_sleeve_configs` yields
   day/swing/longshort from `AI_TRADING_HORIZONS=day:5Min,swing:1Hour,longshort:1Day`.
   `AI_TRADING_SLEEVE_SWING_ENABLED=0` and `AI_TRADING_SLEEVE_LONGSHORT_ENABLED=0`
   remove two of them (the journal shows `"before": ["day"]`), and
   `AI_TRADING_RUNTIME_SLEEVE_WHITELIST=day` with `..._MAX_ACTIVE=1` would remove
   them anyway.
3. It normalizes over the survivors: `0.4 / 0.4 = 1.0` (:38053).

So the stale March file has no effect while only one sleeve is active.
Separately, the journal has no `ALLOCATION_UPDATED`, `ALLOCATION_UPDATE_SKIPPED`
or `ALLOCATION_UPDATE_FAILED` line in 14 days, even though
`AI_TRADING_ALLOCATION_UPDATE_SCHEDULE=market_close`. The updater
(`_maybe_update_allocation_state`, :39078; schedule check `_allocation_schedule_due`,
:39003) appears never to reach its write. I did not trace why. It does not
affect this question.

`regime_signal_components` in decision records does not record where a score
came from. `_decision_record_config_snapshot` (:41327-41333) copies the static
`_REGIME_SIGNAL_COMPONENTS` table (:17600) for the configured profile
(`conservative`) into every record. That table drives `SignalManager`, which is
used by the legacy `_process_symbols` path, not by the netting score.

## 3. Per-strategy status

| `STRATEGIES` entry | Class | Instantiated live | Called live | Status |
|---|---|---|---|---|
| `cross_sectional_momentum` | `CrossSectionalMomentumStrategy` | yes | no | Wired but gated off by `AI_TRADING_NETTING_ENABLED=1` |
| `multi_factor_quality_value` | `MultiFactorQualityValueStrategy` | yes | no | same |
| `pairs_stat_arb` | `PairsStatArbStrategy` | yes | no | same |
| `pead_event` | `PEADEventStrategy` | yes | no | same |
| `low_beta_defensive` | `LowBetaDefensiveStrategy` | yes | no | same |
| `time_series_momentum_overlay` | `TimeSeriesMomentumOverlayStrategy` | yes | no | same |

All six define `generate(ctx)` and a pure `generate_signals(market_data)`.
There is no flag, sleeve setting or regime that routes them into the netting
pipeline. They are reachable only if netting is turned off
(`AI_TRADING_NETTING_ENABLED=0`). That switches the whole runtime back to the
legacy `_process_symbols` + `run_multi_strategy` path. Decision records, sleeve
gates, the day-sleeve ML model and every netting-era control would also stop
applying. That is not a minimal or safe way to evaluate a strategy, and it is
not recommended. Even on that path, `TRADING__ALLOW_SHORTS=0` matters, because
several strategies emit `sell_short`. I did not trace how the legacy path
handles those signals.

Minor defect, not exercised live: if every requested name were unknown,
`get_strategies` falls back to `REGISTRY.get("momentum")` (:19182). `momentum` is
no longer a registry key, so the list would be empty.

Follow-up (fixed after this audit): `get_strategies` no longer falls back to
`momentum`. When no requested name resolves it logs `STRATEGIES_NONE_RESOLVED`
at ERROR with the unknown names and the registry keys, and returns an empty
list. When the netting pipeline is enabled and strategies load, it logs one
`STRATEGIES_NOT_CONSUMED_UNDER_NETTING` warning naming
`AI_TRADING_NETTING_ENABLED`. `.env.example` now leaves `STRATEGIES` commented
out and recommends leaving it unset under netting.

`pead_event` is not an earnings strategy. It reads no earnings calendar. It
fires on any daily gap of at least 3% with volume of at least 1.8× the 40-day
average and same-sign follow-through (`strategies/pead_event.py`).

## 4. Evidence from the live runtime

Service: `ai-trading.service`, `ExecStart=… python -m ai_trading --interval 85`,
`WorkingDirectory=/home/aiuser/ai-trading-bot`, active since 2026-10-18 04:54 UTC.
The runtime env `/run/ai-trading-bot/ai-trading-runtime.env` has the same
`STRATEGIES`, `AI_TRADING_NETTING_ENABLED=1`, sleeve and whitelist values as `.env`.

Journal, `journalctl -u ai-trading.service --since 2026-10-04` (94,119 lines,
2026-10-04 00:00 → 2026-10-18 05:04 UTC):

| Pattern | Count | Meaning |
|---|---|---|
| `Loaded strategies: CrossSectionalMomentumStrategy, LowBetaDefensiveStrategy, MultiFactorQualityValueStrategy, PEADEventStrategy, PairsStatArbStrategy, TimeSeriesMomentumOverlayStrategy` | 17 | Instantiated at each session start or restart (e.g. 2026-10-05 13:30:24Z) |
| `Strategy PEAD Event Drift (pead_event) initialized …` (and the other five) | 17 each | Constructor logs only |
| `Unknown strategy`, `Failed to instantiate strategy`, `Strategy … failed` | 0 | All six loaded without error |
| `No signals produced this cycle`, `Allocator returned no signals`, `Position holding logic failed` | 0 | `run_multi_strategy_cycle` never ran |
| `RUNTIME_SLEEVE_WHITELIST_APPLIED` (`requested ["day"], before ["day"], after ["day"]`) | 2,072 | Netting path ran every cycle |
| `DAY_SLEEVE_FINALIZED_BATCH` | 2,072 | Day-sleeve proposals |
| `NETTING_CYCLE_SLO` / `NETTING_EVENT_DRIVEN_SKIP_NO_NEW_BARS` | 852 / 1,172 | Netting cycle activity |
| `ALLOCATION_UPDATED` | 0 | Allocation state not refreshed (§2) |

Decision records, `/var/lib/ai-trading-bot/runtime/decision_records.jsonl`
(22,841 records, bar_ts 2026-08-11 → 2026-10-16):

- `config_snapshot.allocation_weights`: `{"day": 1.0}` in 22,840 records and `{}`
  in 1 global-block record.
- `config_snapshot.sleeve_configs` keys: `["day"]` in 22,840 and none in 1.
- Every `strategy_id` field is `"day"` (24,844) or null (4,284). Every `sleeve`
  field is `"day"` (45,833).
- 0 occurrences of any of the six strategy ids or class names, and 0 of `"swing"`
  or `"longshort"`.
- Last two weeks (bar_ts ≥ 2026-10-04): 4,957 records over 10 symbols (AAPL, AMZN,
  MSFT, AMD, NVDA, GOOGL, TSLA, META, SPY, QQQ), all `{"day": 1.0}`, 59 with an
  order.
- The rotated backup `decision_records.jsonl.bak.20261008T155820Z.gz` also has 0
  matches for the six strategy ids.

Conclusion: in the window checked, none of the six strategies produced a signal,
an allocation, a decision-record lineage entry or an order.

## 5. Minimal change to evaluate one multi-day strategy on 2024–2025 data

The cheapest valid test leaves the live pipeline alone. It runs the
**unmodified** strategy class offline through the existing registered-trial
machinery, the same way the September ETF trend study did
(`ai_trading/tools/slower_horizon_study.py`): contract → `register_campaign` →
support audit → data-quality check → `claim_campaign_trial` once → evaluate →
`finish_campaign_trial`. Every strategy exposes a pure
`generate_signals(market_data)`, so a harness can call the exact runtime class on
point-in-time data with no `ctx`, broker or netting dependency.

### Recommended candidate: `cross_sectional_momentum`, long-only

- It is defined at daily resolution: 63-session lookback, top/bottom quantile 0.2,
  `min_universe=6`.
- It needs only closes. `pead_event` needs daily open/close/volume and fires
  rarely on mega-caps. On the three-symbol stock set it would almost certainly
  fail the support threshold (the ETF trend contract required ≥100 selections),
  so it would spend budget for an inconclusive result.
- Score the `buy` (top-bucket) signals only. Paper runs with
  `TRADING__ALLOW_SHORTS=0`, and the ETF contract precedent allowed no shorts.

### Required pieces (no runtime, strategy-class or live-config changes)

1. **Owner authorization (research reset).** Under `config/research_reset.json`
   a new hypothesis needs an amendment like
   `docs/RESEARCH_RESET_AMENDMENT_20261018.md`: a dated amendment doc, a new
   `registered_trials` / `amendments` entry, and its own one-trial budget. The
   pause (`unregistered_search` and `new_models` stay paused) and the Sept 9–Dec 8
   holdout stay unchanged. This is a decision for the owner, not the agent.
2. **Frozen contract**, e.g. `config/cross_sectional_momentum_trial_campaign.json`,
   modeled on `config/slower_horizon_campaign.json`. It records: strategy key
   `cross_sectional_momentum` with constructor defaults (no parameter search);
   universe; development window 2024-01-01 → 2025-12-31; `feed: sip`,
   `adjustment: split`; signal evaluated on the close of session *t* using only
   closes ≤ *t*; entry at the open of *t+1*; a fixed non-overlapping hold
   (H sessions, exit at the open of *t+1+H*); equal-notional long slots; 10 bps
   primary and 6/10/20 stress round-trip cost; controls of cash and equal-weight
   always-long on the same universe and dates; circular block bootstrap; and
   support and acceptance thresholds fixed before any outcome is seen.
3. **Harness**, e.g. `ai_trading/tools/registered_strategy_trial.py` (about
   150 lines, cloned from `slower_horizon_study.py`). It builds daily
   regular-session OHLCV from the verified acquisition. At each rebalance date it
   calls `REGISTRY[contract["strategy"]]().generate_signals({"prices": {sym: closes[:t]}})`,
   keeps the `buy` signals and scores open-to-open markouts net of cost. It reuses
   `register_campaign` / `claim_campaign_trial` / `finish_campaign_trial`,
   `audit_sampling_acquisition`, `validate_training_provenance` and
   `load_historical_bars`, and refuses to read any 2026 data.
4. **Regression tests** (`tests/tools/test_registered_strategy_trial.py`):
   - point-in-time truncation (a future price change does not alter the signal at *t*);
   - non-overlapping holds;
   - cost accounting;
   - claim-once and no retry after claim;
   - holdout dates rejected;
   - the contract hash is stable.
5. **Data and support.** The existing development acquisitions are too narrow.
   `artifacts/stock_development/acquisition.json` has 3 symbols (AAPL, AMZN,
   MSFT), below `min_universe=6`. The ETF set has 6 symbols, but DIA had 243
   incomplete sessions, which was the binding constraint on the September study.
   A 2024–2025 SIP acquisition for at least the 10 runtime symbols (better, a
   broader fixed list chosen by a rule dated before 2024, to limit survivorship
   and selection bias) is needed. Its non-claiming support audit must pass before
   the trial is claimed. Daily bars can be aggregated from minute data. Whether
   `historical_training_backfill` can fetch 1Day bars directly was not verified.
6. **Horizon versus support.** In 2024–2025, about 502 sessions minus the
   63-session lookback leaves about 440 usable sessions. A 21-session (monthly)
   hold gives about 20 decision periods. That is far below the 60-period support
   threshold of the ETF contract and would almost certainly come out
   inconclusive. A 5-session hold gives about 88 periods. The hold length must be
   fixed in the contract before outcomes are seen. It is the main design choice
   for the owner.

### Not minimal, and out of scope until a trial passes

Wiring a strategy into the live netting cycle would need all of the following:

- a new sleeve name accepted by `core/horizons.py:63-76`;
- a base weight in `_allocation_base_weights` (bot_engine.py:38022, day/swing/longshort only);
- a bridge from `StrategySignal` to the `score`/`confidence` input of
  `compute_sleeve_proposal`;
- removing that sleeve from the `day`-only whitelist.

Doing so before the offline evidence exists would bypass the research reset.

---

## Changed
- Added `docs/STRATEGY_WIRING_AUDIT_20261018.md` (this file). No code or config
  changes.

## Regression Coverage
- None. This change is documentation only and records a read-only audit; no
  behavior changed.

## Validation
- `bash scripts/agent_validate_changed.sh --docs-only`: exit 0 ("docs-only change
  detected"; changed file is this document only).

## Runtime Check
- Read-only: `systemctl show ai-trading.service`, the runtime env file,
  `journalctl -u ai-trading.service --since 2026-10-04`, and scans of
  `decision_records.jsonl` and its Oct 8 backup. No service restart, no orders,
  no config edits.

## Residual Risk
- The journal window is 14 days and the decision records cover Aug 11–Oct 16.
  Earlier history was not checked.
- Not traced: why `ALLOCATION_UPDATED` never fires, and how the legacy path
  handles `sell_short` when shorts are disallowed.
- §5 is a proposal. Data sufficiency (universe, completeness) has not been
  audited for any new acquisition.
