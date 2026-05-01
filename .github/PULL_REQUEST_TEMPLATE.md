## Worklog

- Intent:
- Root cause:
- Scope:

## Patchset

- Files changed:
- Runtime behavior changed:
- Docs updated:

## Validation

Required for runtime or library changes:

```bash
./venv/bin/pytest -q
./venv/bin/ruff check
./venv/bin/mypy
bash scripts/typecheck_strict.sh
./venv/bin/python -m py_compile $(git ls-files '*.py')
```

For docs-only changes, state why runtime checks were skipped.

## Risk And Rollback

- Risk:
- Rollback:

## Checklist

- [ ] Followed `AGENTS.md`.
- [ ] Kept the diff surgical.
- [ ] Added or updated tests for behavior changes.
- [ ] Used `ai_trading.config.management` for runtime config access.
- [ ] Preserved `alpaca-py==0.42.1` runtime policy.
- [ ] Avoided shims, raw runtime `print`, and unrelated refactors.
