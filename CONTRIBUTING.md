# Contributing to ai-trading-bot

Thanks for helping keep this trading runtime boring in the best possible way:
small changes, clear risk, and validation before confidence.

## Source Of Truth

Read [AGENTS.md](AGENTS.md) before changing runtime code. It is the
authoritative operating contract for ports, SDK policy, configuration access,
logging, validation, and AI-assisted edits.

## Branch And PR Flow

- Create feature branches from `main`: `fix/<topic>` or `feat/<topic>`.
- Keep PRs surgical and scoped to the reported behavior.
- Include Problem, Solution, Validation, Risk, and Rollback in the PR.
- Update docs when runtime behavior, env vars, ports, or operator commands
  change.

## Runtime Rules

- Runtime targets Python 3.12 and `alpaca-py==0.42.1`.
- Use stdlib `zoneinfo`; do not add `pytz`.
- Read runtime configuration through `ai_trading.config.management`.
- Use structured logging; no raw `print` in runtime code.
- Do not add shims or compatibility facades.
- Keep legacy entrypoints thin and delegated to canonical `ai_trading.*`
  modules.
- Guard optional dependencies with direct `try`/`except ImportError` blocks or
  `importlib.util.find_spec` when needed; keep heavy imports inside function
  scope when practical.
- Catch specific exceptions. Broad exception handling must be justified,
  logged, and kept out of hot-path masking behavior.

## Validation

For runtime or library changes, run and report:

```bash
./venv/bin/pytest -q
./venv/bin/ruff check
./venv/bin/mypy
bash scripts/typecheck_strict.sh
./venv/bin/python -m py_compile $(git ls-files '*.py')
```

During market hours, prefer targeted tests plus live runtime validation unless a
broad test run is explicitly approved.

Docs-only changes may skip runtime validation and should say why.

## Repository Hygiene

- Keep the root focused on current operator and contributor surfaces.
- Put one-off audits, implementation snapshots, and historical reports under
  `docs/archive/` or leave them in git history.
- Do not commit generated runtime data, logs, model artifacts, coverage files,
  caches, private workspace memory, or local env files.
- Use the repo virtualenv as `aiuser`; do not run repo-local commands with
  `sudo`.
