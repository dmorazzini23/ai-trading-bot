.PHONY: init test lint verify test-all contract

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

test: contract
	PYTHONPATH=. pytest -q -n auto --maxfail=1 --disable-warnings

contract:
	python tools/import_contract.py

test-all: test

lint:
	python -m py_compile $(shell git ls-files '*.py')

verify:
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
	chmod +x scripts/quick_verify.sh
	./scripts/quick_verify.sh
