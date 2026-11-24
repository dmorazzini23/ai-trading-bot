# OMS Lifecycle Parity Rollout (Staging -> Production)

## Goal
Enable strict lifecycle parity gating so health and runtime go/no-go fail closed
when live/simulated OMS lifecycle streams diverge.

## Prerequisites
1. SQLAlchemy/psycopg runtime dependencies installed.
2. `DATABASE_URL` points to staging Postgres (or staging SQLite for dry-run).
3. Institutional gate script (`ci/scripts/institutional_gates.sh`) is green on
   the target commit.

## Staging Enablement
1. No staging env overlay is shipped in this repository. Outside tests the
   parity keys already default to on:
   `AI_TRADING_HEALTH_OMS_LIFECYCLE_PARITY_ENABLED`,
   `AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY`,
   `AI_TRADING_RUNTIME_PERF_OMS_LIFECYCLE_PARITY_ENABLED` and
   `AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_OMS_LIFECYCLE_PARITY`
   (falling back to `AI_TRADING_RUNTIME_GONOGO_REQUIRE_OMS_LIFECYCLE_PARITY`).
   Confirm the staging `.env` does not set any of them to `0`.
2. Restart staging services.
3. Verify health gate (packaged service on `:9001`; use `HEALTHCHECK_PORT`
   only for the standalone health app):
   - `curl -sS http://127.0.0.1:9001/healthz | jq '.oms_lifecycle_parity, .ok, .reason'`
4. Verify runtime report gate state. The report writes to stdout, so redirect
   it to a file:
   - `python3 -m ai_trading.tools.runtime_performance_report --json --go-no-go > runtime/runtime_performance_report_latest.json`
   - `jq '.oms_lifecycle_parity, .go_no_go' runtime/runtime_performance_report_latest.json`
5. Verify replay parity fixture:
   - `python3 -m ai_trading.tools.oms_lifecycle_parity_replay --fixture tests/data/oms_lifecycle_parity_fixture.json`

## Promotion Gate
Staging is eligible for production promotion only if all are true:
1. `healthz.ok == true` with `oms_lifecycle_parity.ok == true`.
2. Runtime go/no-go passes with:
   - `require_oms_lifecycle_parity=true`
   - `oms_lifecycle_parity_total_violations=0`
3. Replay harness returns `ok=true` and `mismatch_count=0`.

## Rollback
If parity violations appear after enablement:
1. Set:
   - `AI_TRADING_HEALTH_REQUIRE_OMS_LIFECYCLE_PARITY=0`
   - `AI_TRADING_EXECUTION_RUNTIME_GONOGO_REQUIRE_OMS_LIFECYCLE_PARITY=0`
2. Keep `*_ENABLED=1` so telemetry still records violations.
3. Triage latest mismatches via replay harness comparisons and event streams.
