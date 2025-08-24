# Defaults (configurable in CI)
TOP_N ?= 5
FAIL_ON_IMPORT_ERRORS ?= 0
DISABLE_ENV_ASSERT ?= 0
IMPORT_REPAIR_REPORT ?= artifacts/import-repair-report.md

# Ensure artifact dir exists
$(shell mkdir -p artifacts >/dev/null 2>&1)

.PHONY: ensure-runtime test-collect-report ci-smoke

# Install runtime + dev (dev with --no-deps) unless SKIP_INSTALL=1
ensure-runtime:
	@echo "[install] runtime"
	@python -m pip install -r requirements.txt -c constraints.txt
	@echo "[install] dev (no-deps)"
	@python -m pip install -r requirements-dev.txt --no-deps -c constraints.txt

# Collect-only + harvest, always write the artifact, exit 0 or 101
test-collect-report:
	@echo "[collect] pytest --collect-only"
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q --collect-only || true
	@echo "[harvest] $(IMPORT_REPAIR_REPORT)"
	@DISABLE_ENV_ASSERT=$(DISABLE_ENV_ASSERT) TOP_N=$(TOP_N) FAIL_ON_IMPORT_ERRORS=$(FAIL_ON_IMPORT_ERRORS) \
  python tools/harvest_import_errors.py --report $(IMPORT_REPAIR_REPORT)

# CI smoke convenience
ci-smoke:
	@bash tools/ci_smoke.sh
