TOP_N ?= 5
FAIL_ON_IMPORT_ERRORS ?= 0
DISABLE_ENV_ASSERT ?= 0
IMPORT_REPAIR_REPORT ?= artifacts/import-repair-report.md

$(shell mkdir -p artifacts >/dev/null 2>&1)

test-collect-report:
	@echo "[collect] running pytest --collect-only"
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	pytest -q --collect-only || true
	@echo "[harvest] writing $(IMPORT_REPAIR_REPORT)"
	@DISABLE_ENV_ASSERT=$(DISABLE_ENV_ASSERT) \
	TOP_N=$(TOP_N) \
	FAIL_ON_IMPORT_ERRORS=$(FAIL_ON_IMPORT_ERRORS) \
	python tools/harvest_import_errors.py --report $(IMPORT_REPAIR_REPORT) $(if $(FAIL_ON_IMPORT_ERRORS),--fail-on-errors,)

ci-smoke:
	@bash tools/ci_smoke.sh

imports-rewrite:
	python tools/repair_test_imports.py --pkg ai_trading --tests tests \
	  --rewrite-map tools/static_import_rewrites.txt \
	  --report artifacts/import-repair-report.md --write
