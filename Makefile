		# ==== knobs with safe defaults ====
TOP_N ?= 5
FAIL_ON_IMPORT_ERRORS ?= 0
DISABLE_ENV_ASSERT ?= 0
IMPORT_REPAIR_REPORT ?= artifacts/import-repair-report.md
SKIP_INSTALL ?= 0
	
# Ensure artifact dir exists even on CI
$(shell mkdir -p artifacts >/dev/null 2>&1)

.PHONY: ensure-runtime test-collect-report ci-smoke smoke test test-all test-all-heavy lint tests-self lint-core

ensure-runtime:
ifeq ($(SKIP_INSTALL),0)
	python -m pip install -r requirements.txt -c constraints.txt
	python -m pip install -r requirements-dev.txt --no-deps -c constraints.txt
endif

# Collect only + harvest into artifact (always writes report)
test-collect-report: ensure-runtime
	@echo "[collect] running pytest --collect-only"
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	python tools/run_pytest.py --collect-only || true
	@echo "[harvest] writing $(IMPORT_REPAIR_REPORT)"
	@DISABLE_ENV_ASSERT=$(DISABLE_ENV_ASSERT) \
	TOP_N=$(TOP_N) \
	FAIL_ON_IMPORT_ERRORS=$(FAIL_ON_IMPORT_ERRORS) \
	python tools/harvest_import_errors.py --report $(IMPORT_REPAIR_REPORT)

# Print head of artifact and propagate 0/101 from harvester
ci-smoke: test-collect-report
	@echo "=== BEGIN import-repair-report (head -40) ==="
	@head -n 40 $(IMPORT_REPAIR_REPORT) || true
	@echo "=== END import-repair-report ==="

# Deterministic smoke: explicit test files, plugin autoload off
smoke: tests-self
	@echo "== pytest (targeted) =="
	pytest -q \
		tests/test_cli_dry_run.py::test_cli_dry_run_exits_zero_and_marks_indicator \
		tests/test_import_side_effects.py::test_module_imports_without_heavy_stacks \
		tests/test_single_instance_lock.py::test_single_instance_lock_no_sys_exit \
		tests/test_env_validation.py::test_empty_interval_is_handled_gracefully

tests-self:
	@echo "== tools/selftest.sh =="
	@bash tools/selftest.sh

lint-core:
	@echo "== ruff (core) =="
	@ruff check ai_trading/main.py ai_trading/core/bot_engine.py ai_trading/runner.py ai_trading/process_manager.py

.PHONY: scan-extras
scan-extras:
	@echo "[make] strict scan for raw install hints"
	@python tools/scan_extras_hints.py --strict

# Alias for developer convenience
test: smoke

# Run full test suite; disables common auto-loaded plugins for determinism
test-all:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	PYTEST_ADDOPTS="-p no:faulthandler -p no:randomly -p no:cov -m 'not integration and not slow and not requires_credentials'" \
	python tools/run_pytest.py tests

# Run everything, including slow/integration/credentials-marked tests, still without plugin autoload.
test-all-heavy:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	python tools/run_pytest.py tests

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check . || \
	  echo "ruff not installed; skipping lint"
