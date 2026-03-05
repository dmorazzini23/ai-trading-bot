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

Status: Planned.

Primary objective:
- Improve realized net edge by reducing slippage and adverse selection.

Target gates (to be automated in code):
- 30-day realized slippage median improves by at least 10 percent versus baseline.
- Fill-rate at target limit offset does not degrade more than 5 percent.
- Execution drift and reject-rate remain within SLO control bands.
- No increase in stale pending order incidents.

## Phase 3: Live Auto-Demotion + Recovery

Status: Planned.

Primary objective:
- Minimize damage when live regime shifts break model assumptions.

Target gates (to be automated in code):
- Automatic demotion triggers after persistent breaches of calibration or drift SLOs.
- Rollback-to-prior model executes within one cycle without service restart.
- Post-demotion drawdown slope reduces versus pre-demotion window.
- Recovery promotion requires fresh phase gate pass, not stale historical pass.

## Operator Checks

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
