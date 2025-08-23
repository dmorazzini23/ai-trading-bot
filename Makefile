## Central pytest knobs
# AI-AGENT-REF: unify test harness
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTEST_PLUGINS    := -p xdist -p pytest_timeout -p pytest_asyncio
PYTEST_FLAGS      := -q -o log_cli=true -o log_cli_level=INFO
PYTEST_MARK_EXPR  := -m "not integration and not slow"
PYTEST_NODES      := -n auto
TIMEOUT_FLAGS     := --timeout=120 --timeout-method=thread

.PHONY: test-collect test-collect-report test-core test-all repair-test-imports

test-collect:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS) --collect-only

test-core:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS) $(PYTEST_MARK_EXPR) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

test-all:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

test-collect-report:
	# AI-AGENT-REF: generate import error report
	python tools/collect_import_errors.py || true
	@echo "Report written to artifacts/import-repair-report.md"

repair-test-imports:
	bash tools/repair-test-imports.sh
