# CI: Import-error surfacing (quick guide)

This repo prints the **Top-N unique import errors** straight into CI logs and
also writes a Markdown artifact for deeper triage.

- **Make target:** `make test-collect-report`
- **Artifact (default):** `artifacts/import-repair-report.md`
- **Script:** `tools/harvest_import_errors.py` (prepends an Environment header)

---

## Why this exists

Before running the full suite, we do a `pytest --collect-only` to catch import
and packaging problems quickly, then harvest and rank them so you don’t have to
open the artifact to see what broke.

The artifact begins with an environment line (distro, glibc, Python, and wheel
tag) so you always know which combo produced the errors, e.g.:

Environment: Ubuntu 24.04 | glibc 2.39 | CPython 3.12.3 | tag cp312-manylinux_2_39_x86_64

---

## Knobs you can tweak

These are Makefile vars—override them on the command line or in CI.

| Variable | Default | What it does |
|---|---:|---|
| `TOP_N` | `5` | How many unique import errors to print inline in CI logs. |
| `FAIL_ON_IMPORT_ERRORS` | `0` | If `1`, `test-collect-report` exits **101** when any import errors are found (good for gating). |
| `DISABLE_ENV_ASSERT` | `0` | If `1`, skips the strict environment assertion (useful on non-canonical runners). |
| `IMPORT_REPAIR_REPORT` | `artifacts/import-repair-report.md` | Where the Markdown report is written. |

### Examples

Print Top-3 in logs and still succeed:
```bash
make test-collect-report TOP_N=3
```

Fail the job if any import errors are found:

```bash
make test-collect-report FAIL_ON_IMPORT_ERRORS=1
# exit code 101 on error
```

Running on a non-Ubuntu-24.04 host? Disable the env assert:

```bash
make test-collect-report DISABLE_ENV_ASSERT=1
```

Change the artifact path:

```bash
make test-collect-report IMPORT_REPAIR_REPORT=/tmp/collect.md
```

---

What the target actually does (simplified)

```bash
# Called by test-collect-report
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -p xdist -p pytest_timeout -p pytest_asyncio --collect-only || true

DISABLE_ENV_ASSERT=$(DISABLE_ENV_ASSERT) \
python tools/harvest_import_errors.py \
  --top $(TOP_N) \
  $(if $(filter 1,$(FAIL_ON_IMPORT_ERRORS)),--fail-on-errors,) \
  --out $(IMPORT_REPAIR_REPORT)
```

- harvest_import_errors.py normalizes messages like:
  - ModuleNotFoundError: No module named 'pkg'
  - ImportError: cannot import name 'X' from 'pkg'
  - AttributeError: module 'pkg' has no attribute 'Y'

If --fail-on-errors is set and issues exist, the script exits with 101
(so CI can distinguish “import collection failed” from test failures).

---

CI snippet (GitHub Actions)

```yaml
- name: Collect imports and surface Top-N
  run: |
    make test-collect-report TOP_N=5 FAIL_ON_IMPORT_ERRORS=1
```

If your runner isn’t Ubuntu 24.04 / CPython 3.12.3 / glibc 2.39, use:

```yaml
- name: Collect imports and surface Top-N
  run: make test-collect-report DISABLE_ENV_ASSERT=1
```

---

Troubleshooting
  - Mutable default warnings fail pre-commit
    Replace param: dict = {} with param: dict | None = None and set
    param = param or {} (or use dataclasses.field(default_factory=dict)).
  - Harvester says “0 errors” but collection failed
    Ensure pytest --collect-only actually ran and that PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    is set (rogue plugins can hide output).
  - Top-N shows vendor SDKs
    They’re optional by default. Either install extras in CI or mark tests with
    pytest.importorskip("vendor_pkg", reason="optional").

---

Questions or tweaks? Ping the secondary reader (that’s me) and we’ll tune the
harvester/Makefile as the suite evolves.

### Quick knobs (recap)

- `TOP_N` (default 5): how many unique import errors to print in CI logs.
- `FAIL_ON_IMPORT_ERRORS=1`: cause `test-collect-report` to exit with code 101 if any import errors are found.
- `DISABLE_ENV_ASSERT=1`: bypass the environment assertion in the harvester (useful on non-canonical hosts).
- `SKIP_INSTALL=1`: **local only**; skips `pip install` in make targets (CI should not set this).

### Legacy tests
Run `make legacy-mark` to tag tests that still import legacy paths with `@pytest.mark.legacy`.
Core test runs exclude them by default via `-m "not legacy ..."`. When you refactor or delete a legacy test, re-run `make legacy-mark` to keep tags consistent.
