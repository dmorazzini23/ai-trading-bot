# knobs
TOP_N ?= 5
FAIL_ON_IMPORT_ERRORS ?= 0
DISABLE_ENV_ASSERT ?= 0
IMPORT_REPAIR_REPORT ?= artifacts/import-repair-report.md

# ensure artifacts dir exists
$(shell mkdir -p artifacts >/dev/null 2>&1)

.PHONY: ensure-runtime legacy-fix test-collect-report ci-smoke

ensure-runtime:
	python -m pip install -r requirements.txt -c constraints.txt
	python -m pip install -r requirements-dev.txt --no-deps -c constraints.txt

legacy-fix:
	python tools/repair_test_imports.py --pkg ai_trading --tests tests --rewrite-map tools/static_import_rewrites.txt --report artifacts/import-repair-report.md --write || true
	python tools/mark_legacy_tests.py --apply || true

test-collect-report: legacy-fix
	@echo "[collect] running pytest --collect-only"
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	pytest -q --collect-only || true
	@echo "[harvest] writing $(IMPORT_REPAIR_REPORT)"
	@DISABLE_ENV_ASSERT=$(DISABLE_ENV_ASSERT) TOP_N=$(TOP_N) FAIL_ON_IMPORT_ERRORS=$(FAIL_ON_IMPORT_ERRORS) \
	python tools/harvest_import_errors.py --report $(IMPORT_REPAIR_REPORT)

ci-smoke:
	@bash tools/ci_smoke.sh
