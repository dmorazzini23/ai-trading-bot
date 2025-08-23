				## Central pytest knobs
# AI-AGENT-REF: unify test harness
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTEST_PLUGINS ?= -p xdist -p pytest_timeout -p pytest_asyncio
PYTEST_FLAGS_BASE ?= -q -o log_cli=true -o log_cli_level=INFO
PYTEST_MARK_EXPR ?= -m "not legacy and not integration and not slow"
PYTEST_NODES ?= -n auto
TIMEOUT_FLAGS ?= --timeout=120 --timeout-method=thread
WITH_RL ?= 0

# --- Defaults ---
PYTHON ?= python                         # AI-AGENT-REF: default interpreter
PYTEST ?= $(PYTHON) -m pytest
TOP_N ?= 5
FAIL_ON_IMPORT_ERRORS ?=                 # AI-AGENT-REF: optional fail gate
DISABLE_ENV_ASSERT ?=                    # AI-AGENT-REF: optional env assert skip
IMPORT_REPAIR_REPORT ?= artifacts/import-repair-report.md

.PHONY: test-collect extras-rl ensure-runtime ensure-artifacts test-collect-report \
        test-core test-all repair-test-imports legacy-scan legacy-mark fmt ci-smoke

test-collect:
	$(PYTEST) $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) --collect-only $(PYTEST_MARK_EXPR)

extras-rl:
	@if [ "$(WITH_RL)" = "1" ]; then \
	python -m pip install -r requirements-extras-rl.txt -c constraints.txt ; \
	echo "RL extras installed" ; \
	else \
	echo "RL extras disabled (set WITH_RL=1 to enable)" ; \
        fi # AI-AGENT-REF: optional RL stack

ensure-runtime:
	@if [ -z "$$SKIP_INSTALL" ]; then \
	$(PYTHON) -m pip install -r requirements.txt -c constraints.txt; \
	$(PYTHON) -m pip install -r requirements-dev.txt --no-deps -c constraints.txt; \
	fi # AI-AGENT-REF: gate installs
.PHONY: ensure-artifacts
ensure-artifacts:
	@mkdir -p $(dir $(IMPORT_REPAIR_REPORT))  # AI-AGENT-REF: ensure artifacts dir
.PHONY: test-collect-report
## test-collect-report: ensure runtime deps, run pytest --collect-only, and
## harvest import errors into artifacts/import-repair-report.md with env header.
## Setting SKIP_INSTALL=1 bypasses the ensure-runtime step.
test-collect-report: ensure-runtime ensure-artifacts
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	pytest -p xdist -p pytest_timeout -p pytest_asyncio --collect-only || true
	DISABLE_ENV_ASSERT=$(DISABLE_ENV_ASSERT) \
        $(PYTHON) tools/harvest_import_errors.py --top $(TOP_N) \
        $(if $(FAIL_ON_IMPORT_ERRORS),--fail-on-errors,) \
        --out "$(IMPORT_REPAIR_REPORT)"
	@echo "Import report → $(IMPORT_REPAIR_REPORT)"  # AI-AGENT-REF: robust collector
test-core:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) $(PYTEST_MARK_EXPR) -n auto --timeout=120 --timeout-method=thread

test-all:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) $(PYTEST_MARK_EXPR) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

repair-test-imports:
	bash tools/repair-test-imports.sh

.PHONY: legacy-scan
legacy-scan:
	python tools/check_no_legacy_symbols.py

.PHONY: legacy-mark
legacy-mark:
	$(PYTHON) tools/mark_legacy_tests.py --apply  # AI-AGENT-REF: tag legacy tests

.PHONY: fmt
fmt:
	$(PYTHON) -m ruff check --select I --fix .  # AI-AGENT-REF: import order
	$(PYTHON) -m black .

.PHONY: ci-smoke
ci-smoke:
	bash tools/ci_smoke.sh  # AI-AGENT-REF: CI smoke helper
