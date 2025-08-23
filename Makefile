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
TOP_N ?= 8
FAIL_ON_IMPORT_ERRORS ?=

.PHONY: test-collect extras-rl ensure-runtime test-collect-report test-core test-all repair-test-imports

test-collect:
	$(PYTEST) $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) --collect-only

extras-rl:
	@if [ "$(WITH_RL)" = "1" ]; then \
	        python -m pip install -r requirements-extras-rl.txt -c constraints.txt ; \
	        echo "RL extras installed" ; \
	else \
	        echo "RL extras disabled (set WITH_RL=1 to enable)" ; \
        fi # AI-AGENT-REF: optional RL stack

ensure-runtime:
	@$(PYTHON) - <<'PY'\
import importlib, subprocess, sys\
mods = ['pydantic','pytest','pytest_asyncio','pytest_timeout','xdist']\
try:\
    [importlib.import_module(m) for m in mods]\
except Exception:\
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-c', 'constraints.txt'])\
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt', '--no-deps', '-c', 'constraints.txt'])\
PY

.PHONY: test-collect-report
## test-collect-report: ensure runtime deps, run pytest --collect-only, and
## harvest import errors into artifacts/import-repair-report.md with env header.
## Setting SKIP_INSTALL=1 bypasses the ensure-runtime step.
test-collect-report:
	@if [ -z "$$SKIP_INSTALL" ]; then $(MAKE) ensure-runtime; fi
	@PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) --collect-only || true
	@$(PYTHON) tools/harvest_import_errors.py --out "$(REPORT)" --top $(TOP_N) $(if $(FAIL_ON_IMPORT_ERRORS),--fail-on-errors,)
	@echo "Wrote $(REPORT)"

test-core:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) $(PYTEST_MARK_EXPR) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

test-all:
	pytest $(PYTEST_PLUGINS) $(PYTEST_FLAGS_BASE) $(PYTEST_NODES) $(TIMEOUT_FLAGS)

repair-test-imports:
	bash tools/repair-test-imports.sh
