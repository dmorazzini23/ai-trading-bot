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
4. Check environment diagnostics:
   `curl -s http://127.0.0.1:9001/diag | jq .`

### Operator Mutation Auth

All `/operator/*` endpoints now fail closed unless in-code auth is configured.

- Required:
  `AI_TRADING_OPERATOR_TOKEN_MAP`
  Example:
  `{"ops@example.com":"<token-1>","approver@example.com":"<token-2>"}`
- Optional scoped allowlists:
  - `AI_TRADING_OPERATOR_READERS`
  - `AI_TRADING_OPERATOR_OVERRIDE_OPERATORS`
  - `AI_TRADING_OPERATOR_APPROVERS`
  - `AI_TRADING_OPERATOR_ROLLBACK_OPERATORS`
- Requests must send:
  - `Authorization: Bearer <operator-specific-token>`
  - `X-AI-Trading-Operator-Id: <operator-id>`

Do not rely on network placement alone for `/operator/control-plane`,
`/operator/control-plane/manual-overrides`, `/operator/governance`, or the
mutating governance routes.

### Live Durability Policy

Live mode now requires one authoritative durability path:

- `AI_TRADING_OMS_INTENT_STORE_ENABLED=1`
- `DATABASE_URL=postgresql+psycopg://...`
- Live JSONL OMS ledgers are disabled in the hot path.
- Legacy/non-netting live execution is blocked in non-test runtimes.
- Pretrade pacing persists by default at `runtime/pretrade_rate_limiter.db`; point
  `AI_TRADING_PRETRADE_RATE_LIMITER_PATH` at a shared durable runtime volume if
  you run supervised active/passive processes on the same host.
- The legacy SQLAlchemy database manager is retired outside tests; do not use
  the old `trades` / `portfolio` / `risk_metrics` / `performance_metrics`
  schema for runtime durability.

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

Daily automation refreshes evidence and lightweight candidates. Weekly
automation searches horizons, objectives, symbol expansion, exits, and sizing.
Monthly automation performs broader architecture and capital-profile review.
Saturday weekend automation runs bounded broad research/training with a 3-hour
default cap. Sunday weekend automation runs replay/validation/operator synthesis
with a 2-hour default cap and writes `weekend_research_latest.json` plus
`weekend_operator_summary.json` for Monday preparation.
Manual workflows generate gated promotion, live-cutover, incident-replay, and
strategy-change artifacts.

The automation never promotes production models or enables live money. Slack and
OpenClaw summaries should read the generated artifacts instead of running heavy
training directly. See
[docs/RESEARCH_AUTOMATION.md](/home/aiuser/ai-trading-bot/docs/RESEARCH_AUTOMATION.md)
for commands and timer installation.
Research dry-runs intentionally skip Slack/OpenClaw completion notifications.
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

### Provider Safe-Mode

Repeated Alpaca failures can trigger provider safe-mode:

1. A provider outage alert is emitted.
2. The halt flag is written to `HALT_FLAG_PATH`/`AI_TRADING_HALT_FLAG_PATH`.
3. The provider monitor backs the provider off.
4. New orders are blocked until the halt condition clears.

Resume only after Alpaca data is healthy again and the halt condition has been
cleared intentionally.

### Robustness Audit

For the current control matrix, scenario drills, and operator cadence, see
[docs/ROBUSTNESS_AUDIT.md](/home/aiuser/ai-trading-bot/docs/ROBUSTNESS_AUDIT.md).
