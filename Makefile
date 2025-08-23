## Central pytest knobs
# AI-AGENT-REF: unify test harness
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTEST_PLUGINS    := -p xdist -p pytest_timeout -p pytest_asyncio
PYTEST_FLAGS_BASE := -q -o log_cli=true -o log_cli_level=INFO
PYTEST_MARK_EXPR  := -m "not integration and not slow"
PYTEST_NODES      := -n auto
TIMEOUT_FLAGS     := --timeout=120 --timeout-method=thread

.PHONY: test-collect test-core test-all repair-test-imports

test-collect:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) --collect-only

test-core:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) $(PYTEST_MARK_EXPR) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

test-all:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

repair-test-imports:
	bash tools/repair-test-imports.sh
