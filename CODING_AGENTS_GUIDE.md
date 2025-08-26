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
- Include a Validation section: `py_compile`, service restart, journal check.

## Example Commit Message

Thread runtime in regime detection; remove final ctx usage; warn-once for missing tickers

## Example Validation

python -m py_compile $(git ls-files ‘*.py’) || exit 1
sudo systemctl restart ai-trading.service
journalctl -u ai-trading.service -f | sed -n ‘1,200p’

## Known Pitfalls
- “Loaded DataFrame is empty…” off-hours is normal.
- If `MODEL_LOAD_FAILED`, emit `SKIP_TRADE_NO_MODEL` and continue the loop.
