## Central pytest knobs
# AI-AGENT-REF: unify test harness
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTEST_PLUGINS    = -p xdist -p pytest_timeout -p pytest_asyncio
PYTEST_FLAGS_BASE = -q -o log_cli=true -o log_cli_level=INFO
PYTEST_MARK_EXPR  = -m "not integration and not slow"
PYTEST_NODES      = -n auto
TIMEOUT_FLAGS     = --timeout=120 --timeout-method=thread
WITH_RL ?= 0

# Use the active shell's Python (typically your venv) everywhere.
PYTHON ?= $(shell readlink -f $$(command -v python))
PYTEST  = $(PYTHON) -m pytest
REPORT ?= artifacts/import-repair-report.md

.PHONY: test-collect extras-rl test-collect-report test-core test-all repair-test-imports

test-collect:
	$(PYTEST) $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) --collect-only

extras-rl:
	@if [ "$(WITH_RL)" = "1" ]; then \
	        python -m pip install -r requirements-extras-rl.txt -c constraints.txt ; \
	        echo "RL extras installed" ; \
	else \
	        echo "RL extras disabled (set WITH_RL=1 to enable)" ; \
	fi # AI-AGENT-REF: optional RL stack

.PHONY: test-collect-report
## test-collect-report: run pytest --collect-only to surface import errors,
## then harvest them into artifacts/import-repair-report.md.
## The harvester prepends a normalized environment line and
## asserts the exact combo on Ubuntu 24.04 / CPython 3.12.3.
test-collect-report:
	$(PYTEST) $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) --collect-only || true
	# Prepend env header + assert canonical combo; writes $(REPORT)
	IMPORT_REPAIR_REPORT=$(REPORT) $(PYTHON) tools/harvest_import_errors.py
	@echo "Wrote $(REPORT)"

test-core:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) $(PYTEST_MARK_EXPR) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

test-all:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

repair-test-imports:
	bash tools/repair-test-imports.sh
