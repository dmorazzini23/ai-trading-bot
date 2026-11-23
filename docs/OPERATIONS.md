## Operations

Target OS: **Ubuntu 24.04**.

### Service Management

```bash
sudo systemctl start ai-trading.service
sudo systemctl stop ai-trading.service
sudo systemctl restart ai-trading.service
sudo systemctl status ai-trading.service
journalctl -u ai-trading.service -n 200 --no-pager
curl -s http://127.0.0.1:9001/healthz
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9001/metrics
```

The packaged `ai-trading.service` runs the main API on `:9001` and, by
default, exposes `/healthz` and `/metrics` on that same API port.

On restart, packaged services run `scripts/sync_env_runtime.sh` and render
`/run/ai-trading-bot/ai-trading-runtime.env` from the repo `.env`. Operators
should edit `/home/aiuser/ai-trading-bot/.env`; after the updated unit files
are deployed, `sudo systemctl restart ai-trading.service` is enough to refresh
the runtime environment. The sync step fails closed for packaged destinations:
if `/run/ai-trading-bot` is missing or not writable, it exits non-zero instead
of silently writing a repo-local fallback env file.

Use `RUN_HEALTHCHECK=1 python -m ai_trading.app` only when you want the
standalone lightweight health app. In that mode the standalone health server
binds to `HEALTHCHECK_PORT` (default `8081`) and exposes `/healthz`; the
packaged service's metrics endpoint remains on the main API port. In the main
runtime path, `RUN_HEALTHCHECK=1` requires `HEALTHCHECK_PORT != API_PORT`.

### Singleton Guard

The service refuses to start when another healthy `ai-trading` API is already
bound to `API_PORT` (`9001` by default). On startup it probes
`http://127.0.0.1:<API_PORT>/healthz`; if that probe returns
`service=ai-trading`, the new process logs `API_PORT_HEALTHY_ELSEWHERE` and
aborts rather than starting a duplicate trading loop.

### API Port Startup Behavior

`API_PORT_WAIT_SECONDS` defaults to **30 seconds**. Startup waits up to that
window for the API socket to become free. If the port stays busy, the process
exits with status `98`, and the packaged systemd unit uses
`RestartPreventExitStatus=98` so operators can resolve the conflict explicitly.

### Release Identity Check

`packaging/systemd/ai-trading.service` runs `ai_trading.tools.release_identity`
twice as `ExecStartPre`, both times against
`/var/lib/ai-trading-bot/runtime/release_spec.json` with
`--models-root /var/lib/ai-trading-bot/models`:

1. `--pre-migration`, writing
   `/var/lib/ai-trading-bot/runtime/release_identity_pre_migration.json`;
2. `alembic upgrade head`;
3. the normal (post-migration) check, writing
   `/var/lib/ai-trading-bot/runtime/release_identity_preflight.json`.

The tool exits `1` and writes a `blocked` report when any check fails, so
the service does not start. The spec must hold exactly these fields:
`schema_version` (`1`), `tested_commit_sha`, `resolved_config_hash`,
`launch_profile_hash`, `schema_revision`, and `model_artifact` with exactly
`path` and `sha256`. The report's `checks` are:

- `spec_version`, `spec_fields`: the spec shape above.
- `tested_commit`: `tested_commit_sha` is a 40-character SHA equal to the
  checkout's `HEAD`.
- `clean_checkout`: `git status --porcelain` is empty (no modified or
  untracked files).
- `config_hash`, `profile_hash`: the hashes the checkout computes from the
  current config and launch profile equal the spec.
- `schema_revision`: the checkout's single Alembic head equals the spec; the
  pre-migration phase also accepts an older applied revision on that history,
  the normal phase requires the applied revision to equal the spec.
- `model_artifact`: the SHA-256 of `model_artifact.path` (a relative path
  inside `--models-root`) equals `model_artifact.sha256`.
- `model_path_matches_runtime`: that file is the one `AI_TRADING_MODEL_PATH`
  names.

The report's `observed` block lists the values the checkout computed.

`resolved_config_hash` digests a sanitized snapshot of every entry in
`CONFIG_SPECS` (`ai_trading/config/runtime.py`). Any ConfigSpec change, such
as adding, removing or renaming a spec, changes the hash even when `.env` is
unchanged, and startup stays blocked until the spec is updated. Five failed
starts within 300 seconds also trip the unit's start limit
(`StartLimitIntervalSec=300`, `StartLimitBurst=5`).

Safe deploy order (as `aiuser`, outside market hours):

1. Copy the live spec to a scratch file and edit the copy for the candidate.
   From a clean checkout of the candidate commit, with the same environment
   the unit uses (its `Environment=` lines plus
   `/run/ai-trading-bot/ai-trading-runtime.env`), run the tool against the
   copy with `--output` pointing at scratch files, once with
   `--pre-migration` and once without:

   ```bash
   cd <candidate-checkout>
   PYTHONPATH=<candidate-checkout> /home/aiuser/ai-trading-bot/venv/bin/python \
     -m ai_trading.tools.release_identity --pre-migration \
     --spec <scratch>/release_spec.json \
     --models-root /var/lib/ai-trading-bot/models \
     --output <scratch>/pre_migration.json
   PYTHONPATH=<candidate-checkout> /home/aiuser/ai-trading-bot/venv/bin/python \
     -m ai_trading.tools.release_identity \
     --spec <scratch>/release_spec.json \
     --models-root /var/lib/ai-trading-bot/models \
     --output <scratch>/preflight.json
   ```

   The venv is an editable install of `/home/aiuser/ai-trading-bot`.
   `PYTHONPATH` makes modules that exist in the candidate load from it, but a
   module the candidate deleted still resolves from the running checkout.
   Confirm the report's `observed.git_commit_hash` is the candidate commit,
   and set `AI_TRADING_DATA_DIR` to a scratch directory so the run writes
   nothing under `/var/lib/ai-trading-bot`.

   The tool finds the database from `DATABASE_URL` or
   `AI_TRADING_OMS_INTENT_STORE_PATH` (default `runtime/oms_intents.db`,
   relative to the current directory). If neither is absolute in the runtime
   env, export
   `AI_TRADING_OMS_INTENT_STORE_PATH=/home/aiuser/ai-trading-bot/runtime/oms_intents.db`
   (the path the unit uses) for the dry run, and check that
   `observed.schema_revision` is not null.

   If the candidate adds a migration, the normal run can pass
   `schema_revision` only after the migration, which the unit runs between
   the two checks.
2. Back up the live spec, for example
   `cp -p /var/lib/ai-trading-bot/runtime/release_spec.json /var/lib/ai-trading-bot/runtime/release_spec.json.bak`.
3. `sudo systemctl stop ai-trading.service`, so the running process cannot
   load new code or restart against a spec that does not match yet.
4. Check out the candidate commit in `/home/aiuser/ai-trading-bot` and confirm
   `git status --porcelain` is empty.
5. Update the live spec with the reviewed values.
6. `sudo systemctl start ai-trading.service`, then check both report files
   and `curl -sS http://127.0.0.1:9001/healthz`.

Rollback: stop the service, check out the previous commit, restore the spec
backup, start the service. After five failed starts within 300 seconds, run
`sudo systemctl reset-failed ai-trading.service` before starting again.

See [RELEASE_IDENTITY.md](RELEASE_IDENTITY.md) for the reviewed-evidence
requirements.

### Important Paths

- `TRADE_LOG_PATH` defaults to `/var/log/ai-trading-bot/trades.jsonl` unless
  overridden.
- The packaged systemd unit pins `TRADE_LOG_PATH` to
  `/var/log/ai-trading-bot/trades.jsonl`.
- Writable runtime directories are controlled by:
  `AI_TRADING_DATA_DIR`, `AI_TRADING_CACHE_DIR`, `AI_TRADING_LOG_DIR`,
  `AI_TRADING_MODELS_DIR`, and `AI_TRADING_OUTPUT_DIR`.

### Market-Data Incident Checks

1. Confirm the runtime SDK pin:
   `python3 -c "import importlib.metadata as m; print(m.version('alpaca-py'))"`
2. Confirm the current feed config:
   `python3 -c "from ai_trading.config.management import get_env; print(get_env('ALPACA_DATA_FEED'))"`
3. Check the control-plane snapshot:
   `curl -s -H 'Authorization: Bearer <operator-token>' -H 'X-AI-Trading-Operator-Id: <operator-id>' http://127.0.0.1:9001/operator/control-plane | jq .`
4. Check environment diagnostics (operator auth required):
   `curl -s -H 'Authorization: Bearer <operator-token>' -H 'X-AI-Trading-Operator-Id: <operator-id>' http://127.0.0.1:9001/diag | jq .`

### Operator Mutation Auth

`/diag` and all `/operator/*` endpoints fail closed unless in-code auth is
configured.

- Required:
  `AI_TRADING_OPERATOR_TOKEN_MAP`
  Example:
  `{"ops@example.com":"<token-1>","approver@example.com":"<token-2>"}`
  A shared `AI_TRADING_OPERATOR_API_TOKEN` is accepted only when the token
  map is empty and `AI_TRADING_ALLOW_SHARED_OPERATOR_TOKEN` is set, and only
  for read routes
  while `AI_TRADING_OPERATOR_READERS` is empty; it never works for the
  mutating routes.
- Scoped allowlists:
  - `AI_TRADING_OPERATOR_READERS` (optional, read routes)
  - `AI_TRADING_OPERATOR_OVERRIDE_OPERATORS` (required for manual-override
    `POST`/`DELETE`)
  - `AI_TRADING_OPERATOR_APPROVERS` (required for governance approval)
  - `AI_TRADING_OPERATOR_ROLLBACK_OPERATORS` (required for governance rollback)
- Requests must send:
  - `Authorization: Bearer <operator-specific-token>`
  - `X-AI-Trading-Operator-Id: <operator-id>`

Do not rely on network placement alone for `/diag`, `/operator/presets`,
`/operator/plan`, `/operator/control-plane` and its section routes,
`/operator/control-plane/manual-overrides`, `/operator/governance`, or the
mutating governance routes.

`/metrics` uses a separate check: loopback callers are allowed by default
(`AI_TRADING_METRICS_ALLOWED_REMOTE_ADDRS`, default `127.0.0.1,::1`); other
callers need a bearer token matching `AI_TRADING_METRICS_BEARER_TOKEN`, unless
`AI_TRADING_METRICS_ALLOW_UNAUTHENTICATED=1`.

### Live Durability Policy

Live mode now requires one authoritative durability path:

- `AI_TRADING_OMS_INTENT_STORE_ENABLED=1`
- `DATABASE_URL=postgresql+psycopg://...`
- Live JSONL OMS ledgers are disabled in the hot path.
- Legacy/non-netting live execution is blocked in non-test runtimes.
- The legacy non-netting decision cycle has been removed: every trading
  cycle runs the netting pipeline. `AI_TRADING_NETTING_ENABLED` is no longer a
  setting (any value is ignored); remove it from `.env`.
- `STRATEGIES` is no longer read and no strategy classes are loaded at runtime;
  `STRATEGIES_NOT_CONSUMED_UNDER_NETTING` and `STRATEGIES_NONE_RESOLVED` are no
  longer logged. Remove `STRATEGIES` from `.env` if it is still set.
- Pretrade pacing persists by default at `runtime/pretrade_rate_limiter.db`; point
  `AI_TRADING_PRETRADE_RATE_LIMITER_PATH` at a shared durable runtime volume if
  you run supervised active/passive processes on the same host.
- The legacy SQLAlchemy database manager has been removed; do not use
  the old `trades` / `portfolio` / `risk_metrics` / `performance_metrics`
  schema for runtime durability.

### Retired Settings Keys (Phase 4 Cleanup)

The keys below no longer have a reader. The service ignores them: startup
does not reject them, and setting, changing or deleting them has no effect on
trading or on `resolved_config_hash`. They are safe to delete from `.env` after
the phase 4 deploy (then restart so the runtime env file is refreshed).

- Former `TradingConfig` settings: `AI_TRADING_NETTING_ENABLED`,
  `EXECUTION_ALLOW_LAST_CLOSE`, `AI_TRADING_EXEC_ALLOW_LAST_CLOSE`,
  `EXECUTION_ALLOW_FALLBACK_PRICE`, `AI_TRADING_EXEC_ALLOW_FALLBACK_PRICE`,
  `ALLOW_EXECUTION_ON_FALLBACK_QUOTES`, `LOG__EXEC_SUMMARY_ENABLED`,
  `EXECUTION_STALE_RATIO_SHADOW`, `SYMBOL_PROCESS_BUDGET`,
  `AI_TRADING_SIGNAL_HOLD_EPS`, `DEFAULT_MODEL_URL`, `MAX_SYMBOLS_PER_CYCLE`,
  `ORDERS_PENDING_NEW_WARN_S`, `ORDERS_PENDING_NEW_ERROR_S`.
- Former `Settings` fields (unprefixed aliases such as
  `TAKE_PROFIT_EXIT_FRACTION` or `MIN_POSITION_HOLD_SECONDS` are retired too): `AI_TRADING_TAKE_PROFIT_EXIT_FRACTION`,
  `AI_TRADING_TAKE_PROFIT_MIN_HOLD_BYPASS_PCT`, `AI_TRADING_MIN_HOLD_LOSER_CUT_PCT`,
  `AI_TRADING_FALLBACK_ENTRY_CONFIDENCE_BONUS`,
  `AI_TRADING_REVERSAL_EXIT_CONFIRM_SIGNALS`, `AI_TRADING_ENTRY_FLIP_CONFIRM_SIGNALS`,
  `AI_TRADING_FALLBACK_EXPECTED_EDGE_PENALTY_BPS`, `AI_TRADING_ENTRY_COST_BUFFER_BPS`,
  `AI_TRADING_ENTRY_AGE_PENALTY_PER_SEC_BPS`, `AI_TRADING_ENTRY_AGE_PENALTY_CAP_BPS`,
  `AI_TRADING_EXPECTANCY_FILTER_ENABLED`, `AI_TRADING_EXPECTANCY_MIN_SAMPLES`,
  `AI_TRADING_EXPECTANCY_WINDOW_TRADES`, `AI_TRADING_EXPECTANCY_MIN_MEAN_PCT`,
  `AI_TRADING_WINNER_BREAK_EVEN_R`, `AI_TRADING_WINNER_TRAILING_TIGHTEN_R`,
  `AI_TRADING_WINNER_TRAILING_ATR_FACTOR`, `DEFAULT_LIQUIDITY_FACTOR`,
  `AI_TRADING_LIVE_STOP_LOSS_BPS` (alias `AI_TRADING_STOP_LOSS_BPS`),
  `AI_TRADING_LIVE_TAKE_PROFIT_BPS` (alias `AI_TRADING_TAKE_PROFIT_BPS`),
  `AI_TRADING_LIVE_TRAILING_STOP_BPS` (alias `AI_TRADING_TRAILING_STOP_BPS`),
  `DATA_SANITIZE_ENABLED`, `CORP_ACTIONS_ENABLED`, `SIZING_ENABLED`,
  `AI_TRADING_MIN_POSITION_HOLD_SECONDS` (alias `REBALANCE_HOLD_SECONDS`),
  `AI_TRADING_MAX_POSITION_HOLD_SECONDS`, `AI_TRADING_MAX_TRADES_PER_DAY`,
  `AI_TRADING_MAX_TRADES_PER_HOUR`.
- Keys that nothing ever read: all `AI_TRADING_PROFITABILITY_GOVERNOR_*`;
  `AI_TRADING_FEED_RELIABILITY_ENABLED`, `_LIVE_MIN_SAMPLES`, `_MIN_SCORE`,
  `_SIZE_FLOOR`, `_THRESHOLD_BONUS_MAX` (`_LOOKBACK`, `_MIN_SAMPLES` and `_PATH`
  are still read); `AI_TRADING_PAPER_SAMPLING_MAX_TRADES_PER_HOUR`,
  `AI_TRADING_PAPER_SAMPLING_MAX_TRADES_PER_SYMBOL_PER_HOUR`,
  `AI_TRADING_PAPER_SAMPLING_TRADE_COOLDOWN_MIN`.
- Retired in the follow-up cleanup (same rules: ignored at startup, safe to
  delete from `.env` once that deploy is live):
  - Former `TradingConfig` settings, whose removal changed
    `resolved_config_hash`: `API_HOST` (the API and health servers always
    bind `0.0.0.0`), `ORDER_FILL_RATE_TARGET`, `LIQUIDITY_SPREAD_THRESHOLD`,
    `LIQUIDITY_VOL_THRESHOLD`, `LIQUIDITY_REDUCTION_AGGRESSIVE`,
    `LIQUIDITY_REDUCTION_MODERATE` (the execution engine's liquidity
    reduction is hardcoded), `SENTIMENT_ENHANCED_CACHING`,
    `SENTIMENT_FALLBACK_SOURCES`, `SENTIMENT_SUCCESS_RATE_TARGET`,
    `SENTIMENT_RECOVERY_TIMEOUT_SECS`, `META_LEARNING_BOOTSTRAP_ENABLED`,
    `META_LEARNING_BOOTSTRAP_WIN_RATE`, `META_LEARNING_MIN_TRADES_REDUCED`.
    An out-of-range value for one of these used to fail startup; it is now
    ignored.
  - Former `Settings` fields: `AI_TRADING_FINNHUB_RPM`,
    `AI_TRADING_DATA_CACHE_ENABLE`, `AI_TRADING_DATA_CACHE_TTL_SECONDS`,
    `AI_TRADING_DATA_CACHE_DIR`, `AI_TRADING_DATA_CACHE_DISK_ENABLE`,
    `ENV_IMPORT_GUARD`, `AI_TRADING_ENABLE_PLOTTING`,
    `REBALANCE_SLEEP_SECONDS`, `DATA_WARMUP_LOOKBACK_DAYS`, `REGIME_MIN_ROWS`,
    `EXECUTION_PREFER_LIMIT`, `EXECUTION_LOG_SLIPPAGE`,
    `EXECUTION_MAX_PARTICIPATION_RATE` (the per-order participation cap comes
    from `PARTICIPATION_RATE`; order slicing separately reads
    `AI_TRADING_EXEC_MAX_PARTICIPATION_RATE`, which is unaffected),
    `REDIS_URL` (still redacted in logs).
  - Still read, although their duplicate `Settings` field was removed:
    `FORCE_TRADES`, `WEBHOOK_SECRET`, `CPU_ONLY`, `ENABLE_FINNHUB`.

**These keys do not protect positions.** `AI_TRADING_LIVE_STOP_LOSS_BPS`,
`AI_TRADING_LIVE_TAKE_PROFIT_BPS`, `AI_TRADING_LIVE_TRAILING_STOP_BPS`,
`AI_TRADING_MIN_POSITION_HOLD_SECONDS`, `AI_TRADING_MAX_POSITION_HOLD_SECONDS`,
the `AI_TRADING_WINNER_*` keys and the `AI_TRADING_EXPECTANCY_*` keys were read
only by the removed non-netting cycle, which the netting service does not run.
Stop losses and maximum holds are enforced by the position backstop, whose
limits are module constants in `ai_trading/core/netting_position_backstop.py`
(`POSITION_STOP_LOSS_PCT`, `POSITION_MAX_HOLD_SESSIONS`; see "Position
Backstop" below), not by any env key. `AI_TRADING_MAX_TRADES_PER_DAY` and
`AI_TRADING_MAX_TRADES_PER_HOUR` enforce no trade-count limit. Order-count
limits that are still enforced come from other settings, for example
`EXECUTION_MAX_NEW_ORDERS_PER_CYCLE` and, when paper sampling is enabled, the
`AI_TRADING_PAPER_SAMPLING_*` limits that are still read (such as
`AI_TRADING_PAPER_SAMPLING_MAX_TRADES_PER_DAY`).

### Canonical Topology

- Production ownership belongs to `ai-trading.service` only.

### Confirm-First Ops Scripts

- `scripts/runtime_artifacts_reset.sh` defaults to a dry-run plan. Pass
  `--confirm` to archive and rewrite runtime artifacts.
- `scripts/rollout_advanced_gates.sh <stage>` defaults to a dry-run plan. Pass
  `--confirm` to edit `.env`; `--restart` and `--verify` are skipped unless the
  stage is confirmed.
- The Makefile does not create `artifacts/` at parse time. Targets that write
  reports create their artifact directories at execution time.

### Model Artifact Loading

- Generic pickle/cloudpickle/dill model deserialization is retired from the
  supported runtime/model-registry path.
- Use JSON-safe inline artifacts or explicit approved model artifact paths
  instead of generic Python object deserialization.
- Runtime joblib model loads now pass through the shared manifest/checksum gate,
  including symbol-model loading and regime-model initialization in live mode.

### Model Promotion Gate

Before promoting a challenger model, generate a promotion report:

```bash
./venv/bin/python -m ai_trading.tools.promotion_pipeline \
  --model-path /path/to/candidate.joblib \
  --manifest-path /path/to/candidate.joblib.manifest.json \
  --full-replay-json artifacts/replay/full.json \
  --tail-replay-json artifacts/replay/tail.json \
  --recent-replay-json artifacts/replay/recent.json \
  --shadow-report-json artifacts/ml_shadow/latest.json \
  --output-json artifacts/promotion/promotion_report_latest.json
```

The report does not mutate runtime paths. It verifies the checksum manifest and
requires positive full/tail/recent replay, zero replay violations, acceptable
shadow telemetry, live-cost health, and runtime decay controls before reporting
`promotion_ready=true`. Keep the current champion model path available and use
the report rollback command if live behavior regresses.

### Research Automation

Recurring research is orchestrated by
`ai_trading.tools.research_automation` and the
`ai-trading-research-{daily,weekly,monthly,weekend-saturday,weekend-sunday}.timer`
units. The timers write
dated bundles under `/var/lib/ai-trading-bot/runtime/research_reports/` and
stable latest summaries under
`/var/lib/ai-trading-bot/runtime/research_reports/latest/`.
For daily runs, `daily_readiness_latest.json` is the canonical operator answer;
`daily_research_latest.json` is a compatibility alias. The automation report
itself is `daily_research_automation_latest.json` and is the only latest pointer
used to decide whether a completion notification describes the current run.

While the research reset in `config/research_reset.json` is enabled (it is),
`build_research_steps` in `ai_trading.tools.research_automation` replaces
every scheduled cadence, and the manual `strategy-change` workflow, with three
reset steps: `broker_accounting_evidence`, `paper_evidence_review` and
`research_reset_scorecard`. The cadence descriptions below apply only when
the reset policy is not active.

Daily automation refreshes evidence and lightweight candidates. Weekly
automation searches horizons, objectives, symbol expansion, exits, and sizing.
Monthly automation performs broader architecture and capital-profile review.
Saturday weekend automation runs bounded broad research/training with a 3-hour
default cap. Sunday weekend automation runs replay/validation/operator synthesis
with a 2-hour default cap and writes `weekend_research_latest.json` plus
`weekend_operator_summary.json` for Monday preparation.
Manual workflows generate gated promotion, live-cutover, incident-replay, and
strategy-change artifacts.

The automation never promotes production models or enables live money. Slack
summaries should read the generated artifacts instead of running heavy
training directly. See
[docs/RESEARCH_AUTOMATION.md](/home/aiuser/ai-trading-bot/docs/RESEARCH_AUTOMATION.md)
for commands and timer installation.
Research dry-runs intentionally skip Slack completion notifications.
An evidence no-go exits with status `2` and is treated as a successful oneshot
timer result for operator visibility; lock contention exits `75`, and
infrastructure failures exit `1`. Notification delivery failures are logged by
the wrapper without changing the research run exit code.

### Governed Evidence Collection

Diagnostic order collection is restricted to `AAPL,AMZN,MSFT` and remains
paper-only. The runtime reserves capacity across those symbols and the opening,
midday, and closing periods before it applies the existing global, symbol,
side, session, hourly, notional, quote, risk, and OMS caps. Collection reports
may observe other symbols, but ungoverned symbols are never actionable routing
priorities.

Every eligible decision receives a deterministic `correlation_id` before order
submission. That identifier is carried through the quote snapshot, intent,
broker order, fill or non-fill outcome, exit, TCA, calibration, and daily
reports. Treat `metadata_quality.status != complete` or any `ambiguous_*` join
diagnostic as an evidence-quality incident; do not manually substitute
time-proximity joins.

Universal opportunity markouts are research-only. The daily evidence path
records one-, three-, and five-bar outcomes for submitted, non-submitted, and
controlled-skip opportunities. Each row is explicitly marked
`evidence_type=shadow_counterfactual`, `fill_based_evidence=false`, and
`promotion_eligible=false`. Censored session-boundary or missing-bar outcomes
remain visible instead of being silently dropped.

Bounded passive repricing is opt-in with
`AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_ENABLED=1`. It applies only to
locally verified diagnostic entries with a correlation ID and reservation
token. A stale order may be canceled and replaced at the current best passive
quote only when quote-age, spread, retry, cooldown, session, near-close,
notional, risk, and OMS checks pass. The replacement uses a deterministic child
client-order ID, remains a DAY limit, and never crosses the spread. Live mode,
ungoverned symbols, missing metadata, stale or locked quotes, exhausted retries,
and the end-of-day window fail closed to cancellation.

Relevant conservative defaults are documented in `.env.example`:

- `AI_TRADING_PAPER_SAMPLING_STRATIFIED_FAIRNESS_ENABLED=1`
- `AI_TRADING_PAPER_SAMPLING_MAX_TRADES_PER_REGIME_PER_DAY=4` prevents one
  observed market regime from consuming the daily diagnostic budget.
- `AI_TRADING_PAPER_SAMPLING_RESERVED_{OPENING,MIDDAY,CLOSING}_TRADES_PER_DAY=1`
- `AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_ENABLED=0`
- `AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_TIMEOUT_SEC=45`
- `AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_MAX_RETRIES=2`
- `AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_COOLDOWN_SEC=30`
- `AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_QUOTE_MAX_AGE_MS=2500`
- `AI_TRADING_PAPER_SAMPLING_PASSIVE_REPRICE_MAX_SPREAD_BPS=20`

Historical one-minute backfill is a separate, after-hours research path. It
accepts only the governed symbols, records Alpaca feed/adjustment/SDK and
corporate-action provenance, checkpoints bounded session windows, verifies
content hashes and regular-session completeness, and does not interpolate
missing bars. Replay-aligned training applies the live cost model and contiguous
walk-forward folds with an embargo. Historical rows are always tagged
`evidence_type=historical_research`, `promotion_eligible=false`, and
`runtime_fill_authority=false`.

Replay-aligned candidates use positive-class probability as a development-only
ranking score. Percentile selection and all tuning occur inside the development
partition; the frozen probability cutoff is then evaluated once on holdout only
when the candidate has positive, stable, sufficiently supported post-cost
walk-forward evidence. With no development-eligible candidate, holdout and
replay remain unconsumed. The primary accelerator accepts only AAPL, AMZN, and
MSFT, and a requested live-cost model must report ready/usable rather than
silently falling back to static costs.

Drift baselines use two immutable artifacts. First generate a
`model_data_drift_baseline_proposal` with `--baseline-id`; it remains
`review_required`. After review, approve that saved proposal in a separate
invocation with `--approve-proposal-json`, `--approved-by`, and a new output
path. Automation may normalize current evidence and run the monitor, but it
never overwrites or automatically approves a baseline.

The daily report's promotion-eligible sample count is a positive allowlist of
executed paper/live fill evidence. Historical and shadow counts are reported in
separate partitions and cannot satisfy fill-count or model-promotion gates,
even if their input rows contain fill-shaped or realized-PnL fields.

Useful non-mutating checks:

```bash
./venv/bin/python -m ai_trading.tools.research_automation daily --plan-only
./venv/bin/python -m ai_trading.tools.opportunity_markout_report --help
./venv/bin/python -m ai_trading.tools.historical_training_backfill --help
jq . /var/lib/ai-trading-bot/runtime/paper_sampling_state_latest.json
jq . /var/lib/ai-trading-bot/runtime/research_reports/latest/daily_readiness_latest.json
```

Paper fill rates remain conservative diagnostics. Alpaca paper trading does not
model market impact, latency slippage, or limit-order queue position; see the
[Alpaca paper-trading specification](https://docs.alpaca.markets/us/docs/paper-trading).

### Launch Profiles And Live-Readiness

Runtime policy is explicit through `AI_TRADING_LAUNCH_PROFILE`:

- `paper_observe`: no new orders; research and health observation only.
- `paper_trade`: paper trading with normal paper safeguards.
- `live_canary`: tiny live-capital profile, no shorts by default, allowlisted
  symbols only, strict quote/spread caps, and at most three entries per day.
- `live_restricted`: larger but still constrained live profile after canary
  evidence is reviewed.
- `live_normal`: normal live profile, still gated by promotion, provider, cost,
  decay, and daily-loss controls.

Paper remains the default runtime mode. The normal progression is manual:

```text
paper_trade -> live_canary -> live_restricted -> live_normal
```

When `AI_TRADING_REQUIRE_ML_MODEL=1`, the `paper_trade` profile may use the
newest contract-compatible, verified governed shadow model only when
`AI_TRADING_PAPER_ALLOW_SHADOW_MODEL=1`. This authority is paper-only and
exists to collect the execution evidence needed for promotion. Live profiles
never accept shadow governance and continue to require a verified production
artifact. Do not promote a shadow candidate merely to restore trading; it must
pass the normal offline, replay, and runtime evidence gates.

Set `AI_TRADING_HEALTH_REQUIRE_DAY_SLEEVE_MODEL=1` when the day sleeve is a
required runtime dependency. `/healthz` then reports the governed artifact
under `day_sleeve_model` with its model ID, governance authority, training
timestamp, age, and maximum age. Missing, stale, or unverifiable artifacts make
readiness unhealthy; they must be repaired by training and governance, not by
editing timestamps or relaxing freshness and promotion gates.

A failed replay/live parity gate does not stop paper or simulation evidence
collection. Market data, model inference, decision journaling, and paper
evaluation continue, and the replay-blocked decision source is retained as
shadow evidence. In live mode, the same failure blocks only orders that would
increase exposure. Position reductions, exits, broker reconciliation, and
order-state synchronization continue so the safety gate cannot trap existing
exposure.

When runtime go/no-go enforcement is enabled in paper mode, promotion-quality
failures (including replay/live parity) are soft-enforced: orders remain
passive-only and are quantity-scaled so the service can collect new execution
evidence. Integrity failures such as broker-position reconciliation still fail
closed. Live mode never applies this paper softening and remains blocked until
all required promotion gates pass.

The live capital ramp is applied only after all live-readiness gates pass. Its
default phases are `0.25,0.50,0.75,1.00`; each phase multiplies otherwise
eligible live sizing while the capital cap and launch-profile limits continue
to apply. `AI_TRADING_CANARY_PERCENT` controls deterministic symbol sampling,
not the percentage of capital allocated to canary trading.

The active profile and provider-authority decision are exposed in `/healthz` as
`launch_profile` and `provider_authority`. The daily research timer also writes
`live_capital_readiness.json`; live money must remain disabled unless that
artifact and the operator runbook both allow the next step.

Useful manual checks:

```bash
./venv/bin/python -m ai_trading.tools.daily_research_pipeline \
  --output-json /var/lib/ai-trading-bot/runtime/research_reports/latest/daily_readiness_latest.json

./venv/bin/python -m ai_trading.tools.live_capital_readiness \
  --output-json /var/lib/ai-trading-bot/runtime/live_capital_readiness_latest.json \
  --success-on-blocked
```

For live profiles, execution quote authority defaults to Alpaca-only and backup
providers are research-only. Yahoo/backfill data may support research reports,
but it must not silently justify live execution entries.

#### Manual cutover and verification

Do not change `EXECUTION_MODE=paper` or `AI_TRADING_LAUNCH_PROFILE=paper_trade`
until the current live-capital readiness artifact allows the next profile and
the required operator approval has been recorded. Profile changes are never
automatic. After an approved edit to `.env`, restart the packaged service so
systemd refreshes its runtime environment, then verify the service, health,
broker state, and recent decisions:

```bash
sudo systemctl restart ai-trading.service
systemctl status ai-trading.service --no-pager
curl -sS http://127.0.0.1:9001/healthz
journalctl -u ai-trading.service --since '-10 minutes' --no-pager
```

For the first approved live step, use `EXECUTION_MODE=live` with
`AI_TRADING_LAUNCH_PROFILE=live_canary`. Advance to `live_restricted` and then
`live_normal` only after a fresh readiness review for each step. A passing
replay gate alone is not live-capital authority; promotion, provider, cost,
decay, daily-loss, account-confirmation, and manual-approval controls still
apply.

#### Rollback

To remove live authority, restore `EXECUTION_MODE=paper` and
`AI_TRADING_LAUNCH_PROFILE=paper_trade` in `.env`, restart the service, and run
the verification commands above. If an active incident requires an immediate
halt, use the existing kill switch or halt flag first; do not depend on a
configuration restart to manage open exposure. Confirm that broker
reconciliation completes and review open orders and positions before clearing
the incident.

### Health and Control-Plane Signals

When investigating a degraded runtime, check:

1. `/healthz` top-level fields:
   `status`, `ok`, `reason`, `service_state`, and `attention_flags`
2. `/operator/control-plane` top-level fields:
   `rollout`, `service_state`, `attention_flags`, `broker_health`, and `data_provider`

The `drawdown_breaker` section shows the drawdown circuit breaker's in-memory
state (`state` closed/open/unknown, `peak_equity`, `peak_set_at`,
`current_drawdown`, `last_reading_at`, `trip_threshold`, `tripped_at`) and how
its saved state was handled (`state_load` of `restored`, `fresh`, `pending`,
`failed` or `not_configured`, `persisted_state_loaded`, `load_failure_reason`,
`last_write_failed`). `state` is `unknown` until the first equity reading of a
market cycle, because the saved file is only read then: after a weekend restart
it does not say `closed` for a breaker that was tripped. The section is
informational: it never changes `ok`, `status` or the HTTP code.
`{"available": false, "reason": "not_initialized"}` is normal in the main
service until the bot context is first built, and always so in the standalone
health app (`RUN_HEALTHCHECK=1`), which is a separate process. See
`API_DOCUMENTATION.md` for field semantics.

The shared health payload also exposes `operating_mode`. When its mode is
`paper_diagnostics_only`, the configured paper diagnostic path is enabled but
the required governed model remains unavailable. Diagnostic trades are not
qualified-model performance. This explanation does not clear
`required_model_stale`, change HTTP readiness, authorize promotion, or bypass
other trading gates. The market-close recap includes the same explanation.

Important `attention_flags` include:

- `service_degraded`
- `service_halt_active`
- `trade_updates_stream_degraded`
- `provider_backup_active`
- `provider_safe_mode`
- `replay_live_parity_gate_failed`
- `database_unhealthy`
- `oms_invariants_failed`
- `oms_lifecycle_parity_failed`

The non-sending incident snapshot also exposes `operational_state`. It labels
the observation for triage; `/healthz` readiness and the runtime exposure gates
remain authoritative. The labels are evaluated in priority order, so a broker
problem is not hidden by a stale model:

| State | Meaning | Operator response |
| --- | --- | --- |
| `healthy_abstention` | Healthy service, fresh broker, and an explicit closed-market or no-signal reason. | Continue observation; do not force trades. |
| `blocked_qualification` | Model, replay, governance or promotion evidence blocks qualification. | Keep live new exposure blocked; repair evidence through governed processes. |
| `degraded_data` | Provider safe mode, backup, stale or unexplained warm-up. | Check provider freshness and the existing halt/data gates before new exposure. |
| `uncertain_broker_state` | Broker is stale, disconnected or unavailable. | Reconcile positions and open orders; do not infer flat exposure. |
| `execution_incident` | OMS invariants, lifecycle parity or a halt flag failed. | Stop new exposure, preserve order identities and reconcile before recovery. |
| `monitoring_unavailable` | The independent connector could not obtain a valid health payload. | Inspect service and host locally; treat readiness as unknown until the probe recovers. |
| `degraded_unclassified` | Health is degraded without one of the above diagnoses. | Investigate the raw health fields; do not classify it as healthy abstention. |

The existing connector timer is a separate systemd process that probes the
application each minute. As recorded on September 23, 2026, it was enabled
on the host while the older 30-minute healthcheck timer was disabled; check the
current state with `systemctl list-timers 'ai-trading*'`. This detects application failure
while the host and timer remain alive; it cannot detect loss of the host itself.
The state label enriches snapshots and incident text without changing
notification destinations or alert triggers. When triggers clear, the Slack
helper records a local `resolved_at` and retains the prior incident, so a later
recurrence can alert again. The `acknowledge_incident` helper records the exact
active incident signature and operator in that local state; it does not send a
message, suppress future alerts or prove the operator received a notification.
An unconfirmed health-unavailable probe is cleared when health recovers or a
different incident state is observed, so a later outage must earn its own
second confirmation.
Incident notification, acknowledgement and clear operations now share a
same-host file lock and atomic state writes. This prevents an operator's local
acknowledgement from being overwritten by a concurrent connector run; it is
not a distributed acknowledgement system or proof of Slack delivery.
Call `acknowledge_incident` with the reported `incident_signature` and an
`operator` value. Confirm actual recovery
from fresh health, broker and order evidence before treating the incident as
closed. Off-host heartbeat detection, delivery approval and demonstrated
end-to-end acknowledgement remain prerequisites for unattended live operation.

#### Read-only pre-open acceptance verdict

Before market open, run the acceptance gate from the repository as `aiuser`:

```bash
./venv/bin/python scripts/pre_open_acceptance_gate.py --json
```

The `preopen_readiness_verdict` step is a non-mutating six-factor check of the
governed production day/`ml_edge` artifact and its training age, recent shadow
opportunity-markout evidence (96-hour default bounded window), the last 500
decision records within a 4 MiB read bound for `BAD_DATA_CONTRACT`, broker and
order/position snapshot freshness, zero open orders plus the configured
flat-start policy, and coherent paper/simulation authority with
`live_new_exposure_allowed=false`. It reuses the single `/healthz` payload
already fetched by the command. Market-closed provider warming and overnight
signal age are intentionally not blockers; artifact training time and bounded
evidence time are the applicable freshness checks.

Exit codes retain the existing automation contract:

- `0`: every step passed.
- `1`: at least one blocking failure; do not treat the bot as open-ready.
- `2`: no blocking failure, but at least one warning requires operator review.

The default command synchronizes runtime environment state and refreshes
reports before evaluating them. For a strictly read-only diagnostic that does
not perform those preparatory writes, use:

```bash
./venv/bin/python scripts/pre_open_acceptance_gate.py --no-sync-env --no-refresh --json
```

The verdict itself never promotes or deploys a model, submits or cancels an
order, changes exposure, changes launch authority, or relaxes replay/go-no-go
gates. Replay and runtime go/no-go remain independent required steps in the
same report.

### Provider Safe-Mode

Repeated Alpaca failures can trigger provider safe-mode:

1. A provider outage alert is emitted.
2. The halt flag is written to `HALT_FLAG_PATH`/`AI_TRADING_HALT_FLAG_PATH`.
3. The provider monitor backs the provider off.
4. New orders are blocked until the halt condition clears.

Resume only after Alpaca data is healthy again and the halt condition has been
cleared intentionally.

### Drawdown Circuit Breaker

Each netting cycle that reaches the decision stage feeds the account equity
from the execution engine's per-cycle account snapshot (no extra broker call)
to the drawdown breaker.

- **Trip:** equity at or below `(1 - MAX_DRAWDOWN_THRESHOLD)` of the highest
  equity recorded (8% with `MAX_DRAWDOWN_THRESHOLD=0.08`), including peaks
  from before a restart. Logs `DRAWDOWN_BREAKER_TRIPPED` (critical, with
  `peak_set_at`).
- **While open:** orders that increase exposure are rejected with gate
  `DRAWDOWN_BREAKER_BLOCK` in the decision record; orders that reduce a
  position still go out; a reversal has already been cut to its exit leg
  (`FLAT_BEFORE_REVERSAL`) before this check, so it goes out as a reduction.
  It does not set the global halt.
- **Reset:** automatic once drawdown is back at or below 0.8 x the trip level
  (6.4%). Logs `DRAWDOWN_BREAKER_RESET` (warning). The peak only rises.
- **Unusable equity** (no snapshot, missing, non-numeric, zero or negative)
  is never scored: `DRAWDOWN_BREAKER_FEED_SKIPPED` (warning) and the state is
  left as it was.
- **State survives restarts.** Peak equity, when it was set, open/closed, the
  trip time and the broker account id are kept in
  `runtime/drawdown_breaker_state.json` under the data dir
  (`/var/lib/ai-trading-bot` for the packaged service), written atomically
  only when one of them changes. The first equity reading after a start
  restores it, so a drawdown spread over several restarts trips and a restart
  does not close an open breaker. A stored peak never expires.
- **Bad or missing file.** A missing, corrupt or implausible file (wrong
  `schema_version`, non-finite or non-positive peak, future or zone-less
  timestamp) or one for a different account id logs
  `DRAWDOWN_BREAKER_STATE_LOAD_FAILED` (warning, with `reason`), starts fresh
  from the first reading and is overwritten at the next change; it never
  blocks trading or aborts a cycle. A file that exists but cannot be read
  (permissions, I/O error; `reason=unreadable`, logged once a day) is not
  overwritten: the breaker runs in memory, writes nothing, and retries the load
  at every reading. When it loads, the higher peak wins and the breaker is open
  if either side was open. A failed write logs
  `DRAWDOWN_BREAKER_STATE_WRITE_FAILED` and is retried at the next equity
  reading, also while the breaker stays open.
- **What each restart shows.** Closed: `DRAWDOWN_BREAKER_STATE_RESTORED`
  (info, with `peak_equity`, `peak_set_at`, `peak_age_days`). Open:
  `DRAWDOWN_BREAKER_RESTORED_OPEN` (critical, `reason=restored_from_state`,
  with peak, equity, drawdown and `tripped_at`) on every start while it stays
  open. Journal alerts treat it as a trading-stop marker like
  `DRAWDOWN_BREAKER_TRIPPED` ("Trading stopped"). Entries then show
  `DRAWDOWN_BREAKER_BLOCK` as before.
- **A tripped breaker can stay open indefinitely.** While open the bot is
  reduce-only. Once it is flat, equity stops moving, drawdown cannot recover to
  the 6.4% reset level, and entries stay blocked across restarts. Only the
  operator clears that, deliberately:

  ```bash
  sudo systemctl stop ai-trading.service
  sudo rm /var/lib/ai-trading-bot/runtime/drawdown_breaker_state.json
  sudo systemctl start ai-trading.service
  ```

  The first equity reading after the start becomes the new peak and the
  breaker is closed (one `DRAWDOWN_BREAKER_STATE_LOAD_FAILED reason=missing`
  warning). Do the same to re-base after a withdrawal. Do not edit or delete
  the file while the service runs; it is rewritten from memory.
- **Deposits and withdrawals are not detected.** A withdrawal reads as
  drawdown, and a deposit raises the peak.
- **Manual runs touch the real file.** The path resolves under
  `AI_TRADING_DATA_DIR`, then `STATE_DIRECTORY`, then the default data dir
  `/var/lib/ai-trading-bot`. A manual or ad-hoc run of the bot runtime that
  feeds equity with no data dir set reads and writes the real state file; set
  `AI_TRADING_DATA_DIR` to a temporary directory for such runs. A breaker
  constructed without a state path (tests, `TradingHaltManager`) touches no
  file. `peak_equity.txt` is not used.

### Position Backstop (Stop Loss, Maximum Hold, Position Cap)

Safety backstops in the netting cycle, not strategy rules. Code:
`ai_trading/core/netting_position_backstop.py` (named constants, no env keys).

- **Stop loss:** a held position is exited when the cycle's netting price (last
  finalized bar close) is 5% against the broker's `avg_entry_price`
  (`POSITION_STOP_LOSS_PCT`, shorts symmetric). The entry price comes from the
  execution engine's broker-sync snapshot; with no fresh snapshot (600 s), a
  quantity that differs from the cycle's position, or no usable price the
  symbol is unarmed for the cycle (`POSITION_BACKSTOP_UNARMED`, once per symbol
  and session). A move above 40% from entry is not trusted
  (`POSITION_BACKSTOP_PRICE_SUSPECT`).
- **Maximum hold:** exit after 5 NYSE session closes since the episode opened
  (`POSITION_MAX_HOLD_SESSIONS`; weekends, holidays and early closes counted on
  the NYSE calendar). A position the bot finds without a record starts its
  clock when first seen.
- **Exit and lock:** the decision record gets `POSITION_STOP_LOSS_EXIT` or
  `POSITION_MAX_HOLD_EXIT` and a `position_backstop` snapshot entry; the target
  goes to flat and the order goes through the normal netting submission (no
  resting broker stop). The symbol is locked until the next session open: the
  exit keeps being forced while held, and a flat symbol cannot reopen
  (`POSITION_BACKSTOP_REENTRY_BLOCK`). Markers: `POSITION_STOP_LOSS_BREACH`,
  `POSITION_MAX_HOLD_BREACH`, `POSITION_BACKSTOP_EXIT_STALLED` (still held two
  bars after arming), `POSITION_BACKSTOP_LOCK_EXPIRED`. Per-symbol markers log
  as `MARKER | symbol=SYM` so the runtime log throttle keeps one line per symbol.
- **Exits and the entry-quality gates:** two kinds of exit get the same
  treatment: a protective (backstop) exit, and a full close (an order that
  takes the held position, long or short, exactly to flat, including the exit
  leg of a flip). Whether an order is a full close is decided once, on the
  intended order before any liquidity clip; a full close that is clipped or
  scaled keeps the treatment on what remains, and on a later bar the remainder
  is again a full close of the smaller position. A trim that a later step
  resizes to exactly flat (for example ATTACK-tier upsizing) is still a trim.
  Positions are counted in whole shares, so a position under one share counts
  as flat. A partial trim (a reduction that leaves a position) is gated
  exactly like an entry. Gate code: `exit_override_gate()` in the backstop
  module.
  - **Skipped:** policy ablation block, quarantine block mode, event blackout
    (and the pretrade event flag), adaptive sizing, approval blocks for net edge
    floor, spread, calibration, pacing and stale orders, approval soft
    throttles when every scaling reason is a soft throttle (including one that
    rounds the order to `ZERO_QTY`), capacity throttle, the paper sampling
    floor, and the portfolio optimizer. The exit keeps its full size.
  - **Clipped instead of blocked:** the thin-order dollar cap
    (`LIQ_THIN_MAX_ORDER_BLOCK`), and the participation cap when rolling volume
    is zero or the symbol has no liquidity entry (`LIQ_PARTICIPATION_BLOCK`).
    Both clip to `AI_TRADING_LIQ_THIN_MAX_ORDER_DOLLARS` (at least one share,
    never more than the exit, the whole exit if the cap is 0); the thin rules
    do not scale or clip a volume-capped exit again. Later bars (or the
    backstop lock) exit the rest.
  - **Still applies:** halts (including the daily and weekly loss stop), a
    missing price, the kill switch, bar dedup, the participation cap with
    positive volume (block or scale), the thin-liquidity cost multiplier (not
    on a volume-capped exit), approval hard risk blocks (`RISK_*_HARD_BLOCK`),
    any other approval scaling reason, `RISK_CAP_SYMBOL` (symbol notional cap
    and minimum quantity), sell-to-available, duplicate intent, broker
    breakers and all other pretrade checks (for example `DERISK_DATA_QUALITY_BLOCK` on a missing
    quote or a degraded-data flag, stale quote, crossed quote, slippage
    ceiling), except as below. The end-of-day flatten is a separate path and
    does not meet these gates.
  - **Pricing (urgent exits):** a protective exit or a full close goes out as
    a marketable DAY limit, never a market order: a sell at the bid less
    10 bps, a buy-to-cover at the ask plus 10 bps, rounded to the tick away
    from the touch. The decision is taken once, before the liquidity clips, so
    a full close later clipped to a partial order stays urgent. The drawdown
    breaker's reduce-only conversion is treated as a full close, not a
    protective exit. The flags count only while the order actually reduces the
    held position; an add, entry or flip carrying them is priced and checked
    as before. Without a usable quote (missing, crossed or stale), or for a
    protective exit whose touch is more than 10% from the last bar close
    (`far_from_bar_close`), the price is the last bar close moved 50 bps the
    same way. A normal full close whose touch is further from the bar close
    than the price collar is held with `PRICE_COLLAR_BLOCK`
    (`basis=touch_vs_bar_close`). Both pretrade checks see this limit, so a
    move from the bar close inside those bounds does not trip the collar; a
    protective exit's limit may sit within the collar of either the quote mid
    or the bar close. `URGENT_EXIT_LIMIT_PRICE | symbol=SYM` logs the price,
    its basis (`quote` or `bar_close_fallback`) and why the quote was not
    used; the decision snapshot keeps it as `urgent_exit_pricing`. Trims, adds
    and entries keep the bar-close limit.
  - **Quote checks:** a protective exit is not held by `STALE_QUOTE_BLOCK`, a
    missing quote (`DERISK_DATA_QUALITY_BLOCK`), the execution-quality gate or
    `SLIPPAGE_CEILING_BLOCK`. A normal full close still meets the stale,
    missing-quote and slippage-ceiling blocks, but skips the execution-quality
    gate (judged against the held position). A crossed or nonsense quote still
    blocks every exit (`QUOTE_SANITY_BLOCK`), as does a degraded-data flag.
    Trims, adds and entries meet every check as before.
  - **Fractional positions:** broker quantities are truncated to whole shares
    (`POSITION_FRACTIONAL_QTY_TRUNCATED` names the symbol); a remainder below
    one share is never exited by the cycle.
  - **Journal:** the original gate name stays in the decision record, followed
    by `PROTECTIVE_EXIT_OVERRIDE` for a backstop exit or
    `CLOSING_EXIT_OVERRIDE` for a full close. A decision carrying
    `CLOSING_EXIT_OVERRIDE` logs `CLOSING_EXIT_OVERRIDE | symbol=SYM` at INFO
    (no alert).
- **Flips:** the cycle never trades through flat in one order. Right after the
  backstop step, an order that would cross zero is cut to its exit leg (target
  flat, `FLAT_BEFORE_REVERSAL`, `POSITION_REVERSAL_CLAMP` log), so every gate
  judges a full close. The new side is an opening from flat on a later bar and
  meets every entry gate.
- **Backstop specifics:** a move above 40% from entry is treated as bad data so
  the stop loss does not arm. The stall alert
  (`POSITION_BACKSTOP_EXIT_STALLED`) only fires on bars the symbol is
  evaluated; a resting, unfilled exit order keeps the symbol out of the cycle
  (pending-order filter), so it suppresses the stall alert.
- **Position cap:** a new opening is blocked (`POSITION_CAP_BLOCK`) when held
  positions plus flat symbols with pending orders plus openings already sent
  this cycle reach `AI_TRADING_MAX_PORTFOLIO_POSITIONS`. Adds, trims, exits and
  flips are never blocked and nothing is force-closed; if already over the cap,
  `POSITION_CAP_EXCEEDED` logs once per session.
- **State:** `runtime/position_backstop_state.json` under the runtime data dir
  (episodes and locks; written atomically only when it changes). It survives a
  restart; a missing or unreadable file starts empty
  (`POSITION_BACKSTOP_STATE_LOAD_FAILED`) and every held position restarts its
  hold clock. One malformed row drops only that row
  (`POSITION_BACKSTOP_STATE_ROWS_DROPPED`). After a restart a stored episode
  is kept when the broker position has the same side and replaced
  (`first_seen`) when the side differs; a same-side close and reopen while
  the bot was down cannot be detected and keeps the old entry time. To clear
  a lock early, stop the service, delete the symbol's entry under `locks`,
  and start it again.

### Other Packaged Timers And Units

These units also ship under `packaging/systemd/` (schedules from the unit
files; whether each is installed and enabled is host state, so check
`systemctl list-timers 'ai-trading*'`):

- `ai-trading-connectors.timer`: every minute (2 minutes after boot), runs
  `scripts/connector_incident_dispatch.py` (Slack incident connectors).
- `ai-trading-healthcheck.timer`: every 30 minutes, runs
  `scripts/runtime_phase1_health_check_runner.sh`.
- `ai-trading-runtime-report.timer`: Mon..Fri 16:20 America/New_York, runs
  `scripts/refresh_runtime_reports.sh`.
- `ai-trading-replay-governance.timer`: Mon..Fri 05:20 America/New_York and
  20 minutes after boot, runs `ai_trading.tools.replay_governance --force`
  and writes
  `/var/lib/ai-trading-bot/runtime/replay_governance_refresh_latest.json`.
- `ai-trading-host-heartbeat.timer`: every two minutes, runs
  `ai_trading.tools.host_heartbeat` (host liveness to CloudWatch; see
  [INDEPENDENT_HOST_MONITOR.md](INDEPENDENT_HOST_MONITOR.md)).
- `ai-trading-runtime-backup-sync.timer`: daily 23:30 UTC, runs
  `ai_trading.tools.runtime_recovery_backup` then
  `scripts/sync_runtime_backups_to_s3.sh` (see
  [RUNTIME_RECOVERY.md](RUNTIME_RECOVERY.md)).
- `ai-trading-research-manual@.service`: no timer; runs
  `scripts/run_research_automation.sh manual <workflow>` for the instance name.
- `user/ai-trading-evidence-verification.timer` (user unit): fixed dates
  2026-09-07 and 2026-09-08 23:00 UTC, runs
  `ai_trading.tools.scheduled_evidence_verification --session 2026-09-08`.
  Both dates have passed.

### Runtime JSONL retention

`ai-trading-runtime-prune.timer` (every 30 minutes) runs
`scripts/prune_runtime_jsonl.sh`. It archives evidence and never deletes it.

- A live JSONL file above its byte cap keeps the newest 80% of the cap (and at
  most `*_KEEP_LINES` rows). The full file before rotation is hard-linked,
  gzip-compressed and tested at
  `runtime/recovery_backups/archive/jsonl/<name>/<file>.<UTC>.gz`.
- Caps: `decision_records` 192 MiB; `config_snapshots`, `gate_effectiveness`
  and `ml_shadow_predictions` 256 MiB; `order_events`, `fill_events` and
  `tca_records` 128 MiB; `execution_quality_events` and `oms_events` 64 MiB;
  `broker_position_boundaries`, `policy_ablation_events` and
  `trade_updates_stream` 32 MiB;
  `memory_samples` 5 MB. Override with `AI_TRADING_<FILE>_MAX_BYTES` and
  `AI_TRADING_<FILE>_KEEP_LINES`. `ai_trading.tools.runtime_artifact_retention`
  uses the same caps for its plan.
- Research report run directories (`research_reports/{daily,weekly,weekend,monthly}/<UTC>_<kind>`)
  are archived when both the run timestamp and every file are older than
  `AI_TRADING_RESEARCH_REPORT_ARCHIVE_DAYS` (default 30). A verified archive goes to
  `runtime/recovery_backups/archive/research_reports/<cadence>/<run>.tar.gz`,
  and only then is the source directory removed. Runs named in model/registry
  JSON, top-level runtime JSON, small runtime JSONL, recent reports or
  `latest/` stay in place. `latest/`, `huggingface/` (the configured HF cache),
  `manual/` and top-level report files are never moved. At most
  `AI_TRADING_RESEARCH_REPORT_ARCHIVE_MAX_DIRS` (default 25) runs are archived
  per invocation. `AI_TRADING_RESEARCH_REPORT_ARCHIVE_ENABLED=0` disables this step.
- Preview without changes:
  `bash scripts/prune_runtime_jsonl.sh --dry-run` (as `aiuser`, with
  `AI_TRADING_RUNTIME_DIR` and `AI_TRADING_DATA_DIR` set).
- Restore a report run:
  `tar -C runtime/research_reports/<cadence> -xzf <archive>`. Restore a JSONL segment
  with `zcat <archive>`.
- The archive tree is local only. The S3 uploader still sends only the newest
  recovery bundle. Off-host copies of archives need a separately approved
  uploader scope.

### CPU-Only PyTorch In The Runtime Venv

The host has no GPU, so torch is pinned to the `+cpu` build from
`https://download.pytorch.org/whl/cpu` (`requirements.txt`, `constraints.txt`).
A venv built from the lock gets no `nvidia-*`/`triton` wheels. To convert an
existing CUDA venv, do it after the close with the account flat, no open
orders, and the service stopped. Run pip as `aiuser`, never with `sudo`.

```bash
sudo systemctl stop ai-trading.service
cd /home/aiuser/ai-trading-bot
cp -a venv venv.cuda-bak                      # rollback copy
P=./venv/bin/python
V=$($P -c 'import torch; print(torch.__version__.split("+")[0])')
$P -m pip uninstall -y triton $($P -m pip freeze | grep -oE '^nvidia-[^=]+')
$P -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==${V}+cpu"
$P -m pip check
$P -c 'import torch, transformers, alpaca; assert not torch.cuda.is_available(); print(torch.__version__)'
sudo systemctl start ai-trading.service
curl -sS http://127.0.0.1:9001/healthz
```

This keeps every other installed version unchanged. Rollback:
`sudo systemctl stop ai-trading.service && rm -rf venv && mv venv.cuda-bak venv`,
then start the service. Delete `venv.cuda-bak` once a session has run cleanly.

### Security Pin Bumps In The Runtime Venv

The October 2026 dependency audit bumped four pins in `constraints.txt` to
their minimum fixed releases: `anyio==4.14.2`, `fsspec==2026.6.0`,
`urllib3==2.8.0`, and `werkzeug==3.1.9`. The live venv was built earlier and
may still carry older copies (urllib3 was `2.6.3` there, not `2.7.0`). Upgrade
after the close with the account flat, no open orders, and the service
stopped. Run pip as `aiuser`, never with `sudo`.

```bash
sudo systemctl stop ai-trading.service
cd /home/aiuser/ai-trading-bot
cp -a venv venv.bak                           # rollback copy
P=./venv/bin/python
$P -m pip list | grep -iE '^(anyio|fsspec|urllib3|werkzeug) '   # record old versions
$P -m pip check > /tmp/pip-check-before.txt || true                # baseline (venv drift may already show here)
$P -m pip install --no-deps anyio==4.14.2 fsspec==2026.6.0 urllib3==2.8.0 werkzeug==3.1.9
$P -m pip check | diff /tmp/pip-check-before.txt - && echo "no new conflicts"
$P -c 'import anyio, fsspec, urllib3, werkzeug, flask, requests, httpx, aiohttp, alpaca; print(urllib3.__version__)'
sudo systemctl start ai-trading.service
curl -sS http://127.0.0.1:9001/healthz
```

`--no-deps` keeps every other installed version unchanged. Rollback:
`sudo systemctl stop ai-trading.service && rm -rf venv && mv venv.bak venv`,
then start the service. Delete `venv.bak` once a session has run cleanly.

### Robustness Audit

For the current control matrix, scenario drills, and operator cadence, see
[docs/ROBUSTNESS_AUDIT.md](/home/aiuser/ai-trading-bot/docs/ROBUSTNESS_AUDIT.md).

### Research evidence validation

Scheduled daily accelerator runs require validated historical input via
`--require-validated-data`. When the historical workflow is enabled, automation
passes its current-run acquisition manifest to the accelerator. The resolved
dataset directory is used for both training and replay. Missing or failed
completeness evidence blocks the accelerator with
`training_data_completeness_unverified`; it does not fall back to legacy CSVs.
Weekly and weekend accelerator runs use the latest published historical
acquisition manifest and enforce the same validation gate.

Direct local-CSV research remains available, but reports
`quality_status=unverified_completeness`, with per-symbol date coverage,
observed local session dates, cadence, intraday gaps, regime counts, and loader
cleanup counts. Observed dates and median-cadence gaps do not certify exchange
calendar completeness. Use the governed acquisition manifest for that check.

Cost model source diagnostics include `rows_used`, `rows_rejected`, and
`rejection_counts`. Each parsed rejected row has one primary reason; malformed
JSON rows remain in `invalid_rows`. Exact duplicate records within an execution
source are excluded. This does not perform cross-source order/fill joins or
certify correlation integrity.

Accelerator reports and successful cache states carry explicit `candidates`
records for regime evaluation. Walk-forward counts remain research evidence;
they do not create shadow-fill evidence or grant promotion authority. An empty
candidate collection produces `no_candidate_records` rather than treating the
accelerator summary as an unnamed candidate.

### Controlled research and completion evidence

The subsequent focused research cycle is documented in
[FOCUSED_RESEARCH_DECISION.md](FOCUSED_RESEARCH_DECISION.md), with its frozen
scope and acceptance criteria in
[FOCUSED_RESEARCH_PROTOCOL.json](FOCUSED_RESEARCH_PROTOCOL.json). Its tools are
`python -m ai_trading.research.focused_cycle` and
`python -m ai_trading.tools.execution_evidence_reconciliation`; both write
research/audit artifacts and confer no trading or promotion authority.

Scheduled accelerator training recomputes exchange-calendar coverage from the
acquired CSVs before fitting. The defaults require at least 20 observed sessions
per requested symbol and no more than 2% missing regular-session minute bars.
Use `--min-training-sessions` and `--max-training-missing-ratio` to set an explicit
research coverage policy. Reports retain date coverage, missing sessions/bars,
symbol coverage, and past-only regime distributions. Calendar completeness does
not establish balanced regime support; assess the reported distribution before
claiming generalization to a regime. Daily automation uses the latest acquisition
manifest when it is not running a new acquisition.

Training reports include `cost_sensitivity`: frozen out-of-sample selections are
evaluated at `--cost-scenarios-bps` (default `0,3,6,10,20`). Costs are per one-way
execution; an equal-notional round trip has turnover two. Break-even cost is
gross markout divided by turnover. These are research markouts, not portfolio
returns or evidence of queue position or market impact. A fresh, sufficient
quote prior adds a labeled quote-derived stress scenario. Quote observations do
not supply fill-validation authority. With no selected trades, net edge and
break-even cost remain unavailable rather than being reported as profitable.

`--research-experiments` runs feature removal (configured with
`--experiment-feature-removals`), volatile-regime abstention, cash, always-long,
and simple momentum controls on the same outer folds. Feature-removal thresholds
are selected inside the purged inner partition. Reports rank the controls by
net edge per common opportunity and fold stability, recording hypotheses,
acceptance criteria, and paired fold improvements. Horizon/model candidates
retain the existing out-of-sample qualification and holdout controls.

The accelerator passes a stable `--experiment-ledger-json` under its training
cache. Distinct failed evidence signatures accumulate across data refreshes;
`--experiment-max-failures` defaults to two. Already evaluated evidence is not
repeated, and retired experiments are skipped before fitting. Inconclusive
control comparisons do not count toward retirement. A substantive hypothesis
or protocol change must have a distinct experiment contract; changing an output
directory or model display name does not constitute a new hypothesis. Direct
pipeline invocations should explicitly reuse the same ledger path across runs.

Governance writes `loss_attribution` even when non-regression fails. Its
components reconcile the capped-minus-uncapped mean fill markout into retained
order selection, fill mix/timing, and execution-price drag. Separate dollar
markout accounting measures quantity resizing while holding baseline fills
fixed. The comparison uses identical recorded decisions, so entry-selection
change is zero. Linked exits are not present in this metric and are explicitly
reported as not identifiable; zero contribution must not be interpreted as
evidence that exit behavior is harmless. Sequential accounting is sensitive to
component order and includes downstream simulator interactions.

The governance artifact also carries observed replay cost rows for
`ai_trading.tools.replay_live_cost_alignment_report`. The report lists comparison
items even when fresh fill buckets are absent, and requires sufficient fresh
fill-derived evidence for every item before reporting acceptable realism.
`status=ready` indicates report construction succeeded; inspect
`cost_realism.acceptable` for the execution-validation result.

Historical record (September 6, 2026; not the current state). Read-only
verification outputs from this implementation are under
`artifacts/research_completion/`. The September 6, 2026 cost refresh rejects all
7,384 fills and 18,697 TCA rows as older than the three-session window. The
validated historical dataset covers 126 sessions for AAPL, AMZN, and MSFT.
The real training run reaches the registry selector with an identified model;
it remains economically unqualified, rather than failing with a missing model
ID. The governed replay still fails non-regression, but its -0.649113 bps
difference reconciles to -0.371026 bps retained-order effects, +0.079471 bps
fill effects, and -0.357558 bps execution-price effects. Its 3,360 cost comparison
items lack fresh fill validation. These artifacts grant no promotion authority.

Both evaluated horizons also include a 32.582728 bps one-way quote-derived
stress scenario from nine recent sufficient buckets, alongside the five
configured assumptions. Neither selects an out-of-sample trade in this run;
their break-even execution cost is consequently null. The candidate comparison
report now retains `candidate_evaluations` for screened-out and stopped
experiments as well as ranked survivors. Zero measured net edge ranks above
negative edge; missing/non-finite evidence ranks last.

### Scheduled Slack Reports

Two weekday reports post through the Slack incident webhook (hydrated from the
managed secret like the connector timer). They replace the retired OpenClaw cron
jobs; the old Monday runtime audit was not carried over.

- `ai-trading-slack-report-preflight.timer`: 09:20 America/New_York, health and
  readiness from `ai_trading.tools.market_preflight`.
- `ai-trading-slack-report-close-recap.timer`: 16:45 America/New_York,
  `scripts/market_close_recap.py`.

They are systemd **user** units (no sudo; lingering is enabled for `aiuser`):

```bash
mkdir -p ~/.config/systemd/user
cp packaging/systemd/user/ai-trading-slack-report* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ai-trading-slack-report-preflight.timer ai-trading-slack-report-close-recap.timer
systemctl --user list-timers 'ai-trading-slack-report*'
```

Manual send: `systemctl --user start ai-trading-slack-report@preflight.service`.
Non-sending preview: build the text with
`connector_incident_dispatch.build_scheduled_report("preflight")`.


### Journal Slack Alerts

The connector timer (`scripts/connector_incident_dispatch.py`, every minute) also
reads the `ai-trading.service` journal with `journalctl` (the `aiuser` account is
in `systemd-journal`; no sudo) and sends four alert kinds through the same Slack
webhook. Code: `ai_trading/monitoring/journal_alerts.py`, sender
`tools/mcp_slack_alerts_server.py:tool_notify_journal_alerts`.

- **Service lifecycle**: `Started` (INFO, so a manual deploy shows as a start),
  `Failed with result`, non-zero `Main process exited`, `Scheduled restart job`
  (CRITICAL). Failures are held back 10 minutes per type and report a count.
- **Trading stopped / resumed**: markers `DRAWDOWN_BREAKER_TRIPPED`,
  `DISASTER_HALT_TRIGGERED`, `ALERT_HALT_TRADING`, `HALT_FLAG_WRITTEN`,
  `DATA_KILL_SWITCH_ACTIVATED` (stopped) and `DRAWDOWN_BREAKER_RESET`,
  `HALT_FLAG_REMOVED`, `TRANSIENT_HALT_CLEARED` (resumed). 15 minute hold per marker.
- **Position backstop**: `POSITION_STOP_LOSS_BREACH`, `POSITION_MAX_HOLD_BREACH`,
  `POSITION_BACKSTOP_EXIT_STALLED` (WARNING), one message per marker type per
  dispatcher run listing the symbols (a single symbol reads like "Safety
  backstop: AAPL long stop-loss exit, -5.3% vs entry"). 15 minute hold per
  marker and symbol.
- **New error**: each ERROR/CRITICAL log line type (logger plus message with
  numbers collapsed) alerts once, then at most hourly with an occurrence count. The standing `MODEL_REGISTRY_STALE` error repeats at most once a day.

State (journal cursor, per-type last-sent and suppressed counts) is
`slack_journal_alert_state.json` beside the incident state file. The first run
only records the cursor. Disable with `AI_TRADING_CONNECTOR_SLACK_JOURNAL_ENABLED=0`.
Non-sending preview of the last 24h (no webhook, no state written):

```bash
./venv/bin/python scripts/connector_incident_dispatch.py --journal-dry-run 1440
```

`ai-trading.service` has `OnFailure=ai-trading-systemd-failure-alert@%n.service`,
which runs `scripts/systemd_failure_alert.sh` (delegates to
`connector_incident_dispatch.py --unit-failed`) when systemd gives up on the unit.
Install (needs sudo, owner only):

```bash
sudo cp packaging/systemd/ai-trading.service packaging/systemd/ai-trading-systemd-failure-alert@.service /etc/systemd/system/
sudo systemctl daemon-reload   # takes effect at the next service restart
```
