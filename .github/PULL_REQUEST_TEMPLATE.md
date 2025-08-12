<!--
PR Template: Follow all sections. Keep edits surgical; prefer function-level changes over massive rewrites.
-->

# Title
Short, imperative summary (e.g., "Thread runtime through regime detection; remove ctx references")

## Repository Context
- Python 3.12 app on DigitalOcean droplet (systemd-managed)
- Production code under `ai_trading/*`
- House rules:
  - No shims
  - No `try/except ImportError` in prod
  - No broad `except Exception`
  - Structured JSON logging only (no print)
  - Use `runtime` (not `ctx`) across hot paths

## Problem Statement
Explain the exact failure/logs or limitation motivating this change.
- Example: `NameError: model` in `run_all_trades_worker` during model usage.

## Scope of Work
Bullet the concrete edits (files/functions). Avoid scope creep.

## Acceptance Criteria
- [ ] Compiles: `python -m py_compile $(git ls-files '*.py')`
- [ ] Service restarts cleanly: `sudo systemctl restart ai-trading.service`
- [ ] No regressions in logs (attach first 200 lines of `journalctl -u ai-trading.service -f`)
- [ ] No `ctx` reintroduced; `runtime` threaded correctly
- [ ] No shims; no `try/except ImportError`; no `except Exception`

## Change Details
- Files touched:
  - `ai_trading/...`
- Key functions and signatures changed:
  - e.g., `check_market_regime(runtime, state)`
- New helpers (if any):
  - e.g., `_load_primary_model(runtime)`
- Logging:
  - New events (e.g., `MODEL_LOADED`, `MODEL_LOAD_FAILED`, `SKIP_TRADE_NO_MODEL`)

## Constraints & Standards
- Python 3.12 compatibility
- Structured logging preserved
- No destructive refactors without explicit approval

## Implementation Requirements
- Thread `runtime` through new/updated functions
- Cache models on `runtime.model` (no globals)
- Catch specific exceptions only

## Deliverables
- Code diffs
- Updated docs (if applicable)

## Validation Steps
```bash
python -m py_compile $(git ls_files '*.py') || exit 1
sudo systemctl restart ai-trading.service
journalctl -u ai-trading.service -f | sed -n '1,200p'
```

Risk & Rollback
•Risks:
•e.g., model loading failures, runtime threading errors
•Rollback plan:
•git revert <commit>; restart service

Non-Goals
•Strategy re-tuning
•Exception hygiene outside touched surfaces
•Broker routing / ML architecture overhaul / new data providers

Appendices
•Logs/snippets, references to issues/alerts

---

