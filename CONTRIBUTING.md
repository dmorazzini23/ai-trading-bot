# Contributing to ai-trading-bot

## Branch & PR Flow
- Create feature branches from `main`: `fix/<topic>` or `feat/<topic>`.
- Keep PRs surgical; include a clear Problem → Solution → Risk & Rollback section.
- Use our **PR Task Prompt** structure (below) for AI-assisted changes.

## House Rules (Agents & Humans)
- **No shims**. Do not import from `scripts/*` in prod modules.
- **No `try/except ImportError`** in production code. Install dependencies or feature-flag at config.
- **No bare `except Exception:`**. Catch specific exceptions (e.g., `FileNotFoundError`, `ValueError`).
- **Structured logging only**. Use existing JSON logging helpers; do not print().
- **Use `runtime`** (an instance of `BotRuntime`) across hot paths. **Do not introduce `ctx`**.
- Keep `ai_trading` imports stable; avoid dynamic `exec`/`eval`.

## Runtime & Config
- `TradingConfig`: read-only settings (broker creds, paths, limits).
- `BotRuntime`: per-process state: `cfg`, `params`, `tickers`, `model`, etc.
- All hot-path functions must accept `runtime` explicitly.

## Metrics & Imports
- Normalize metrics imports to `ai_trading.monitoring.metrics` (no root-level circulars).
- Prefer explicit imports; avoid wildcard.

## Testing & Validation
- Compile: `python -m py_compile $(git ls-files '*.py')`
- Lint (optional): `ruff check .` / `flake8` if configured.
- Service: `sudo systemctl restart ai-trading.service`
- Logs: `journalctl -u ai-trading.service -f | sed -n '1,200p'`

## Lint & Tests
- Run `make test-all` before committing.
- Outputs are saved under `artifacts/`:
  - `tool-versions.txt` for Python and tool versions
  - `ruff.txt` and `ruff-top-rules.txt` for lint results
  - `mypy.txt` for type-checking
  - `pytest.txt` for test execution

## Keeping test imports current
- Use `tools/repair_test_imports.py` to rewrite stale `ai_trading` imports in tests.
- Run:

  ```
  python tools/repair_test_imports.py --pkg ai_trading --tests tests --write --report artifacts/import-repair-report.md
  ```

- Do not add compatibility shims; tests must reference the real modules.

## PR Task Prompt (required)
**Title**  
**Repository Context**  
**Problem Statement**  
**Scope of Work**  
**Acceptance Criteria**  
**Change Details**  
**Constraints & Standards**  
**Implementation Requirements**  
**Deliverables**  
**Validation Steps**  
**Risk & Rollback**  
**Non-Goals**  
**Appendices**

## Do / Don’t Quicklist
- ✅ Thread `runtime` everywhere; cache the ML model on `runtime.model`.
- ✅ Use specific exceptions; keep logs structured.
- ❌ No global `ctx`.  
- ❌ No shims, no broad `except`, no dynamic exec/eval in prod.


## Runtime settings
- Prefer `AI_TRADING_*` environment variables (e.g., `AI_TRADING_INTERVAL`, `AI_TRADING_MODEL_PATH`).
- Heavy ML features (`torch`, `hmmlearn`) load only when `USE_RL_AGENT=1`.

## Smoke test
Run the lightweight verification script:

```
scripts/smoke.sh
```
