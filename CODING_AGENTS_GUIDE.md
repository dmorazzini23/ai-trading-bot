# Coding Agents Guide (Codex / LLMs)

## Mission
Make precise, minimal edits with first-pass correctness. Do not change strategy math unless explicitly requested.

## Hard Constraints
- **No shims**; do not import from `scripts/*` in production modules.
- **No `try/except ImportError`** in prod; dependencies must be present.
- **No `except Exception:`** catch-alls; use specific exceptions.
- **No global `ctx`**; use `runtime` everywhere (may alias `ctx = runtime` locally inside a single function if needed).
- Preserve **structured JSON logging**.
- Gate heavy libraries (e.g. `pandas`) inside functions; tests importing them must call
  `pytest.importorskip("pandas")`.

## Edit Playbook
1. **Search first**: locate function by name, not line numbers.
2. **Thread `runtime`** through function calls and write to `runtime` fields (e.g., `runtime.tickers`, `runtime.model`).
3. **Model access**: call `_load_primary_model(runtime)` or read `runtime.model`.
4. **Logs**: prefer single “warn-once” for recurring environment issues (e.g., missing tickers file).
5. **No destructive refactors** unless the PR explicitly asks for them.

## Patch Style
- Provide unified diffs or discrete file patches.
- Use `apply_patch` for all file edits.
- Include a Validation section with repo-venv commands. Do not make service restarts routine; restart only when rollout validation explicitly requires it, and call out why.

## Example Commit Message

Thread runtime in regime detection; remove final ctx usage; warn-once for missing tickers

## Example Validation

./venv/bin/python -m py_compile $(git ls-files '*.py')
./venv/bin/pytest -q tests/path/to_targeted_test.py
systemctl status ai-trading.service
curl -sS http://127.0.0.1:9001/healthz
journalctl -u ai-trading.service -n 200 --no-pager

## Known Pitfalls
- “Loaded DataFrame is empty…” off-hours is normal.
- If `MODEL_LOAD_FAILED`, emit `SKIP_TRADE_NO_MODEL` and continue the loop.
