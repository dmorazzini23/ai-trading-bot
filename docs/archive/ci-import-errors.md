# Import-Error Surfacing in CI

- `make test-collect-report` runs `pytest --collect-only`, writes `artifacts/import-repair-report.md`,
  prepends a normalized environment header, and exits **0** when no import errors, **101** otherwise.
- `tools/ci_smoke.sh` (used by `make ci-smoke`) prints the first 40 lines of the artifact and propagates the same exit codes.

## Knobs

- `TOP_N` — number of ranked unique import errors shown in logs (default **5**).
- `FAIL_ON_IMPORT_ERRORS` — set to **1** to fail CI (exit 101) when errors are found.
- `DISABLE_ENV_ASSERT` — set to **1** to bypass the canonical env assertion.
- `SKIP_INSTALL` — set to **1** to skip `pip install` for a fast smoke run.
- `IMPORT_REPAIR_REPORT` — path to the Markdown artifact (default **artifacts/import-repair-report.md**).

