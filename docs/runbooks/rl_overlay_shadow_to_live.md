# RL And Advanced Model Containment

Advanced models are research surfaces until they earn authority through the same
promotion and live-readiness controls as simpler replay-aligned models.

## Default State

- RL stays disabled or shadow-only.
- Execution bandits stay shadow-only unless explicitly promoted.
- Counterfactual learning may refresh evidence, but it must not auto-promote a
  production model.
- Promotion reports must log model type, model path, manifest checksum, replay
  summaries, shadow summaries, cost evidence, and rollback path.

## Promotion Requirements

An advanced model can approach live authority only when all of these are true:

1. Full validation is green.
2. Full, tail, and recent replay windows are acceptable.
3. Shadow telemetry is acceptable under current live quote/cost conditions.
4. Live cost model is ready with no unresolved breaches.
5. Runtime decay controls are healthy.
6. `live_capital_readiness` is not blocked.
7. Operator manually approves the cutover.

## Explicit Non-Goals

- Do not let RL execute live just because it exists.
- Do not let a bandit route live orders unless it has a promotion report.
- Do not hide advanced model decisions from daily reports.
- Do not retire the simpler champion until the challenger has evidence and a
  rollback path.
