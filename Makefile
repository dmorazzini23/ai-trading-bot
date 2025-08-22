.PHONY: init test lint verify test-all contract audit-exceptions self-check deps-dev

init:
	python tools/check_python_version.py
	python -m pip install --upgrade pip setuptools wheel
	# purge any stale installations of this project (editable installs or wheels)
	python -m pip uninstall -y ai-trading-bot || true
	@if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
	# safety net to ensure core test deps even if previous step was interrupted
	python -m pip install "pytest" "pytest-xdist" "tzlocal>=5.2,<6" "psutil>=5.9,<6" "alpaca-trade-api>=3.0,<4" \
	        "pytest-asyncio>=0.23,<0.24" "anyio>=4,<5" "aiohttp>=3.9,<3.10" "websockets>=12,<13"
	# safety net
	python -m pip install "joblib>=1.3,<2"

test: contract
	PYTHONPATH=. pytest --maxfail=100 --disable-warnings --strict-markers -vv

contract:
        python tools/import_contract.py --ci --timeout 20 --modules ai_trading,trade_execution

deps-dev:
        python -m pip install -r requirements.txt -r requirements-dev.txt

test-all: deps-dev
        # Bounded pre-flight import contract; never hang CI
        python tools/import_contract.py --ci --timeout 20 --modules ai_trading,trade_execution
        # Run tests with artifacts; do not stop on first failure
        pytest -n auto --disable-warnings --maxfail=0 --durations=20 \
          --junitxml=artifacts/junit.xml --cov=ai_trading --cov=trade_execution \
          --cov-report=xml:artifacts/coverage.xml -q

lint:
	python -m py_compile $(shell git ls-files '*.py')

verify:
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
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
