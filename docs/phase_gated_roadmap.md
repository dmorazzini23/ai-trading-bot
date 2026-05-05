# Phase-Gated Roadmap

This document tracks the implementation plan for reaching institutional-grade
model governance and execution quality.

Each phase has explicit pass/fail gates. A phase is considered complete only
when all listed gates pass for the defined evaluation window.

## Phase 1 (Week 1): Promotion Hardening + Observable Gates

Status: Implemented in `ai_trading.training.after_hours`.

Primary objective:
- Prevent weak models from being promoted.
- Make promotion decisions auditable and reproducible.

Runtime report location:
- `/var/lib/ai-trading-bot/runtime/research_reports/after_hours_training_YYYYMMDD.json`
- Field: `roadmap.phase_1_week_1`

Phase 1 gates:
- `rows >= AI_TRADING_ROADMAP_PHASE1_MIN_ROWS` (default `1200`)
- `support >= AI_TRADING_ROADMAP_PHASE1_MIN_SUPPORT` (default `120`)
- `mean_expectancy_bps >= AI_TRADING_ROADMAP_PHASE1_MIN_EXPECTANCY_BPS` (default `1.5`)
- `max_drawdown_bps <= AI_TRADING_ROADMAP_PHASE1_MAX_DRAWDOWN_BPS` (default `1800`)
- `turnover_ratio <= AI_TRADING_ROADMAP_PHASE1_MAX_TURNOVER_RATIO` (default `0.35`)
- `hit_rate_stability >= AI_TRADING_ROADMAP_PHASE1_MIN_HIT_RATE_STABILITY` (default `0.60`)
- `brier_score <= AI_TRADING_ROADMAP_PHASE1_MAX_BRIER_SCORE` (default `0.27`)
- `profitable_fold_ratio >= AI_TRADING_ROADMAP_PHASE1_MIN_PROFITABLE_FOLD_RATIO` (default `0.45`)
- `sensitivity_sweep.gate == true`
- Prior-model score delta gate when enabled:
  - `candidate_score - prior_score >= AI_TRADING_ROADMAP_PHASE1_MIN_PRIOR_SCORE_DELTA` (default `0.15`)

Optional promotion enforcement:
- `AI_TRADING_AFTER_HOURS_PROMOTION_REQUIRE_PHASE1_GATE=1`
- When enabled, promotion is blocked unless `roadmap.phase_1_week_1.gate_passed == true`.

## Phase 2: Execution Edge Upgrade (TCA-Native Routing)

Status: Implemented as guarded opt-in automation; pending paper calibration and
baseline refresh before live enablement.

Primary objective:
- Improve realized net edge by reducing slippage and adverse selection.

Target gates:
- 30-day realized slippage median improves by at least 10 percent versus baseline.
- Fill-rate at target limit offset does not degrade more than 5 percent.
- Execution drift and reject-rate remain within SLO control bands.
- No increase in stale pending order incidents.

Current automation surface:
- Smart order routing can consume recent execution-learning context and apply a
  small marketable-limit offset increase when fill rate is below target.
- Routing action is disabled by default and guarded by minimum sample count,
  reject-rate, slippage, max-offset, and side-aware buy/sell directionality.
- Daily execution report now emits `roadmap.phase_2_execution_edge` with windowed
  metrics, thresholds, gate booleans, and `gate_passed`.
- Institutional gate script can enforce this at merge/deploy time via:
  - `AI_TRADING_INSTITUTIONAL_REQUIRE_PHASE2_GATE=1`
  - `AI_TRADING_INSTITUTIONAL_PHASE2_MAX_REPORT_AGE_HOURS` (default `36`)
- Gate thresholds and baselines are runtime-configurable via:
  - `AI_TRADING_ROADMAP_PHASE2_ENABLED`
  - `AI_TRADING_ROADMAP_PHASE2_WINDOW_DAYS`
  - `AI_TRADING_ROADMAP_PHASE2_MIN_SLIPPAGE_IMPROVEMENT_PCT`
  - `AI_TRADING_ROADMAP_PHASE2_MAX_FILL_RATE_DEGRADATION_PCT`
  - `AI_TRADING_ROADMAP_PHASE2_MAX_REJECT_RATE`
  - `AI_TRADING_ROADMAP_PHASE2_MAX_EXECUTION_DRIFT_BPS`
  - `AI_TRADING_ROADMAP_PHASE2_MAX_STALE_PENDING_INCREASE`
  - `AI_TRADING_ROADMAP_PHASE2_BASELINE_SLIPPAGE_MEDIAN_BPS`
  - `AI_TRADING_ROADMAP_PHASE2_BASELINE_FILL_RATE`
  - `AI_TRADING_ROADMAP_PHASE2_BASELINE_STALE_PENDING_COUNT`
  - `AI_TRADING_ROADMAP_PHASE2_BASELINE_PATH` (JSON artifact, default `runtime/phase2_execution_baseline.json`)
- Routing action is controlled separately via:
  - `AI_TRADING_PHASE2_EXECUTION_EDGE_ROUTING_ENABLED` (default off)
  - `AI_TRADING_PHASE2_EXECUTION_EDGE_MIN_SAMPLES`
  - `AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_FILL_RATE`
  - `AI_TRADING_PHASE2_EXECUTION_EDGE_MAX_REJECT_RATE`
  - `AI_TRADING_PHASE2_EXECUTION_EDGE_TARGET_SLIPPAGE_BPS`
  - `AI_TRADING_PHASE2_EXECUTION_EDGE_MAX_OFFSET_ADD_BPS`
  - `AI_TRADING_PHASE2_EXECUTION_EDGE_OFFSET_WEIGHT`

Baseline artifact refresh:

```bash
python3 -m ai_trading.tools.update_phase2_execution_baseline \
  --tca-path runtime/tca_records.jsonl \
  --output-path runtime/phase2_execution_baseline.json \
  --window-days 30
```

## Phase 3: Live Auto-Demotion + Recovery

Status: Runtime scaffolding implemented; rollback remains conservative and must
be paper-drilled before live enablement.

Primary objective:
- Minimize damage when live regime shifts break model assumptions.

Target gates:
- Automatic demotion triggers after persistent breaches of calibration or drift SLOs.
- Rollback-to-prior model executes within one cycle without service restart.
- Post-demotion drawdown slope reduces versus pre-demotion window.
- Recovery promotion requires fresh phase gate pass, not stale historical pass.

Current automation surface:
- Live KPI guard can evaluate control-band breaches on an interval from runtime.
- Breach windows persist to `live_kpi_breach_state.json` under the governance
  base path, or the override path below.
- Consecutive breach requirements are enforced per strategy/model/KPI signature.
- Auto-rollback is dry-run disabled unless explicitly enabled; dry-run audit
  still records failed KPI names, values, current model, breach count, and
  required breach count.
- If rollback is enabled but no previous champion exists, the current production
  model is demoted to `challenger` so stale promotion state is not left marked
  as production.
- Controls:
  - `AI_TRADING_PROMOTION_LIVE_KPI_GUARD_ENABLED`
  - `AI_TRADING_PROMOTION_LIVE_KPI_GUARD_INTERVAL_SEC`
  - `AI_TRADING_PROMOTION_LIVE_KPI_BREACH_STATE_PATH`
  - `AI_TRADING_PROMOTION_LIVE_KPI_BREACH_CONSECUTIVE_REQUIRED`
  - `AI_TRADING_PROMOTION_AUTO_ROLLBACK_ON_CONTROL_BAND` (default off)
- `evaluate_live_kpis_and_maybe_rollback(..., allow_rollback=False)` enables
  breach evaluation without immediate rollback, so callers can enforce
  persistence windows before demotion.

## Operator Checks

## Research Automation Cadence

Daily, weekly, monthly, and manual research workflows are now centralized in
`ai_trading.tools.research_automation`.

- Daily: evidence refresh, live cost model, shadow report, symbol scorecard,
  runtime decay controls, go/no-go snapshot, and lightweight candidate training
  when research bars are configured.
- Weekly: symbol expansion, exit/sizing/objective research through multi-horizon
  replay-aligned candidates and optional microstructure bridge review.
- Monthly: broader regime/model robustness checks and a paper-mode cutover drill
  artifact for capital-profile review.
- Manual: promotion reports, live-cutover drills, incident replay, and major
  strategy-change review.

All automated cadences produce artifacts only. Production promotion, live-money
enablement, incident response, and major strategy changes remain explicit
operator decisions.

Phase 1 verification command:

```bash
latest=$(ls -1t /var/lib/ai-trading-bot/runtime/research_reports/after_hours_training_*.json | head -1)
jq '{ts, promotion: .promotion, phase1: .roadmap.phase_1_week_1}' "$latest"
```

One-command daily runtime guardrail check:

```bash
/home/aiuser/ai-trading-bot/scripts/runtime_phase1_health_check.sh
```

This checks:
- `AUTH_HALT` spike rate in recent decision records.
- `OK_TRADE` collapse in recent decision records.
- required gate-effectiveness artifacts exist and are fresh.
- latest after-hours report freshness.
- shadow predictions artifact freshness when `AI_TRADING_ML_SHADOW_ENABLED=1`.

To reduce false alarms off-hours, the checker automatically suppresses
`AUTH_HALT`/`OK_TRADE` threshold alerts when:
- decision window sample size is below `RATE_ALERT_MIN_ROWS` (default `100`), or
- decision window is stale beyond `DECISION_STALE_MAX_AGE_MINUTES` (default `90`)
  outside regular US hours (`RTH_TZ`, `RTH_START_HHMM`, `RTH_END_HHMM`).

It also suppresses empty-decision-window failures off-hours by default:
- `SUPPRESS_OFFHOURS_EMPTY_DECISION_ALERTS=1`
- `FAIL_ON_EMPTY_DECISION_WINDOW_DURING_RTH=1`
