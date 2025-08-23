## Central pytest knobs
# AI-AGENT-REF: unify test harness
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTEST_PLUGINS := -p xdist -p pytest_timeout -p pytest_asyncio
PYTEST_FLAGS   := -q -o log_cli=true -o log_cli_level=INFO
PYTEST_MARKS   := -m "not integration and not slow"
PYTEST_NODES   := -n auto
PYTEST_TIME    := --timeout=120 --timeout-method=thread

.PHONY: test-collect test-core test-all repair-test-imports

test-collect:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS) --collect-only

test-core:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS) $(PYTEST_MARKS) $(PYTEST_NODES) $(PYTEST_TIME)

test-all:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS) $(PYTEST_NODES) $(PYTEST_TIME)

repair-test-imports:
	bash tools/repair-test-imports.sh

