# Reuse these everywhere to avoid plugin autoload issues
PYTEST_PLUGINS = -p xdist -p pytest_timeout -p pytest_asyncio
PYTEST_OPTS    = -q -o log_cli=true -o log_cli_level=INFO

# --- Pytest config (autload disabled) ---

# Common flags

.PHONY: init test lint verify test-all test-core test-int contract audit-exceptions self-check deps-dev lint-fix lint-fix-phase2 lint-fix-phase3 lint-fix-phase4r lint-histo typecheck

init:
	python tools/check_python_version.py
	python -m pip install --upgrade pip setuptools wheel
	# purge any stale installations of this project (editable installs or wheels)
	python -m pip uninstall -y ai-trading-bot || true
	@if [ -f requirements.txt ]; then pip install -r requirements.txt -c constraints.txt; fi
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt --no-deps -c constraints.txt; fi
	# safety net to ensure core test deps even if previous step was interrupted
	python -m pip install "pytest" "pytest-xdist" "tzlocal>=5.2,<6" "psutil>=5.9,<6" "alpaca-trade-api>=3.0,<4" \
	        "pytest-asyncio>=0.23,<0.24" "anyio>=4,<5" "aiohttp>=3.9,<3.10" "websockets>=12,<13"
	# safety net
	python -m pip install "joblib>=1.3,<2"

test: contract
	PYTHONPATH=. pytest --maxfail=100 --disable-warnings --strict-markers -vv

contract:
	python tools/import_contract.py --ci --timeout 20 --modules ai_trading,trade_execution

dev-deps:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt -c constraints.txt
	@if [ -f requirements-dev.txt ]; then python -m pip install -r requirements-dev.txt --no-deps -c constraints.txt; fi
	python -m pip install -e .
.PHONY: test-core test-int test-all test-core-seq test-core-1p test-collect test-debug repair-test-imports

test-core:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest $(PYTEST_PLUGINS) $(PYTEST_OPTS) -m "not integration and not slow" -n 2 --timeout=120 --timeout-method=thread

.PHONY: test-core-1p
test-core-1p:
	@mkdir -p artifacts
	# AI-AGENT-REF: single process verbose debug
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest \
	        -p pytest_asyncio -p pytest_timeout -p no:xdist \
	        -m "not integration and not slow" \
	        -vv -s --maxfail=1 \
	        | tee artifacts/pytest-core-1p.txt

.PHONY: test-collect
test-collect:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest $(PYTEST_PLUGINS) $(PYTEST_OPTS) --collect-only tests

.PHONY: test-debug
test-debug:
	@mkdir -p artifacts
	# AI-AGENT-REF: trace active plugins/config
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest \
	        -p xdist -p pytest_asyncio -p pytest_timeout \
	        --trace-config -q \
	        | tee artifacts/pytest-trace-config.txt

test-int:
	@mkdir -p artifacts
	# AI-AGENT-REF: load xdist and asyncio when autoload disabled
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RUN_INTEGRATION=1 pytest -p xdist -p pytest_asyncio -n auto -q -m "integration" --disable-warnings | tee artifacts/pytest-integration.txt

test-all:  ## Run lint, types, and unit tests
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt -c constraints.txt
	# Ensure any old stubs that demand urllib3>=2 are gone
	python -m pip uninstall -y types-requests urllib3-stubs || true
	# Install dev tools without pulling transitive deps that override runtime pins
	python -m pip install -r requirements-dev.txt --no-deps -c constraints.txt
	python -m pip install -e .
	$(MAKE) test-core
	@echo "--- Integration (opt-in) ---"
	@RUN_INTEGRATION=${RUN_INTEGRATION} $(MAKE) -s test-int || true
test-core-seq:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q -m "not integration and not slow" --disable-warnings | tee artifacts/pytest-core-seq.txt
.PHONY: lint-fix
lint-fix:
	tools/lint_safe_fix.sh

.PHONY: lint-fix-phase2
lint-fix-phase2:
	tools/lint_safe_fix.sh  # AI-AGENT-REF: phase2b lint pass

.PHONY: lint-fix-phase3
lint-fix-phase3:
	       bash tools/lint_phase3.sh

.PHONY: lint-fix-phase4r
lint-fix-phase4r:
	@mkdir -p artifacts
	@ruff check . --fix --select F401,F841,UP,DTZ,T201 --ignore E501 \
	| tee artifacts/ruff-passA.txt || true
	@python tools/codemods/none_comparisons.py || true
	@python tools/codemods/logger_prints.py || true
	@ruff check . --fix --select F401,F841,UP,DTZ,T201 --ignore E501 \
	| tee artifacts/ruff.txt || true
	@python tools/ruff_histogram.py < artifacts/ruff.txt > artifacts/ruff-top-rules.txt
	@grep -E '^[^:]+:[0-9]+:[0-9]+: [A-Z]+[0-9]{3}' artifacts/ruff.txt | wc -l > artifacts/ruff-count.txt  # AI-AGENT-REF: count lint issues

.PHONY: lint-histo
lint-histo:
	python tools/ruff_histogram.py artifacts/ruff.txt > artifacts/ruff-histogram.md || true

.PHONY: lint
lint:
	ruff check .

.PHONY: typecheck
typecheck:
	python -m mypy --version | tee artifacts/mypy-version.txt
	python -m mypy ai_trading trade_execution | tee artifacts/mypy-phase3.txt || true  # AI-AGENT-REF: ensure type safety

verify:
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt --no-deps -c constraints.txt; fi
	chmod +x scripts/quick_verify.sh
	./scripts/quick_verify.sh
audit-exceptions:
	python tools/audit_exceptions.py --paths ai_trading --fail-over 300

self-check:
	python -m ai_trading.scripts.self_check

# === Import-time config hygiene helpers ===

scan-import-time:
	python tools/scan_import_time.py || true

fix-import-time:
	python tools/fix_import_time.py

refactor-config-hygiene: scan-import-time fix-import-time scan-import-time

repair-test-imports:
	bash tools/repair-test-imports.sh
