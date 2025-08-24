# ---- knobs ----
TOP_N ?= 5
FAIL_ON_IMPORT_ERRORS ?= 0
DISABLE_ENV_ASSERT ?= 0
IMPORT_REPAIR_REPORT ?= artifacts/import-repair-report.md

# ensure artifacts dir exists at parse time (no-op if present)
$(shell mkdir -p artifacts >/dev/null 2>&1)

.PHONY: ensure-runtime test-collect-report ci-smoke

# smoke-friendly runtime bootstrap (idempotent)
ensure-runtime:  # AI-AGENT-REF: quiet install for CI smoke
	@python -m pip install -r requirements.txt -c constraints.txt -q
	@python -m pip install -r requirements-dev.txt --no-deps -c constraints.txt -q || true

# collect + harvest (always produce a report; never fail before write)
test-collect-report:  # AI-AGENT-REF: deterministic artifact generation
	@echo "[collect] pytest --collect-only"
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q --collect-only || true
	@echo "[harvest] -> $(IMPORT_REPAIR_REPORT)"
	@DISABLE_ENV_ASSERT=$(DISABLE_ENV_ASSERT) TOP_N=$(TOP_N) FAIL_ON_IMPORT_ERRORS=$(FAIL_ON_IMPORT_ERRORS) \
	python tools/harvest_import_errors.py --report $(IMPORT_REPAIR_REPORT)

ci-smoke:
	@bash tools/ci_smoke.sh
