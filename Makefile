# --- knobs (centralized) ---
TOP_N ?= 5
FAIL_ON_IMPORT_ERRORS ?= 0
DISABLE_ENV_ASSERT ?=
IMPORT_REPAIR_REPORT ?= artifacts/import-repair-report.md

# Use the active shell's Python/venv everywhere:
PY := $(shell command -v python || echo python)

# Ensure artifacts dir exists before any writes
ARTIFACTS_DIR := artifacts
$(ARTIFACTS_DIR):
	@mkdir -p $(ARTIFACTS_DIR)

# Install runtime/dev (skippable)
ensure-runtime:
ifndef SKIP_INSTALL
	$(PY) -m pip install -r requirements.txt -c constraints.txt
	$(PY) -m pip install -r requirements-dev.txt --no-deps -c constraints.txt
endif

# Collect-only + harvest to artifact (always prints env header)
test-collect-report: $(ARTIFACTS_DIR) ensure-runtime
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -p xdist -p pytest_timeout -p pytest_asyncio \
	    --collect-only || true
	@TOP_N=$(TOP_N) DISABLE_ENV_ASSERT=$(DISABLE_ENV_ASSERT) \
	  $(PY) tools/harvest_import_errors.py --report "$(IMPORT_REPAIR_REPORT)" \
	  --top $(TOP_N) $(if $(FAIL_ON_IMPORT_ERRORS),--fail-on-errors,)

# Quick legacy scan (optional)
legacy-scan:
	@$(PY) tools/check_no_legacy_symbols.py
