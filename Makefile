# Install dependencies
PY ?= $(shell command -v python3 || echo python)
PIP := $(PY) -m pip
PYTEST := PYTHONPATH=. pytest --disable-warnings

install:
	$(PY) -m pip install --upgrade pip
	$(PIP) install -e .

install-dev: install
	$(PIP) install -e .[dev]

# Environment validation
validate-env:
	$(PY) scripts/validate_test_environment.py

# Testing targets
# Back-compat target: only marked tests (unit+integration)
test-marked: clean install-dev validate-env
	$(PYTEST) -q -m "unit or integration"
	@echo "🔎 Checking for legacy trade_execution imports..."
	@if grep -rn "^from trade_execution\|^import trade_execution" --include="*.py" . ; then \
	  echo "❌ Found legacy 'trade_execution' imports. Please migrate to 'from ai_trading import ExecutionEngine'."; \
	  exit 1; \
	else \
	  echo "✅ No legacy 'trade_execution' imports found."; \
	fi

.PHONY: test-all test-fast test-e2e test-marked

# Run EVERYTHING (no markers)
test-all: clean install-dev validate-env
	$(PYTEST) -q
	@echo "🔎 Checking for legacy trade_execution imports..."
	@if grep -rn "^from trade_execution\|^import trade_execution" --include="*.py" . ; then \
	  echo "❌ Found legacy 'trade_execution' imports. Please migrate to 'from ai_trading import ExecutionEngine'."; \
	  exit 1; \
	else \
	  echo "✅ No legacy 'trade_execution' imports found."; \
	fi

# Fast inner loop
test-fast: clean install-dev validate-env
	$(PYTEST) -q -m "unit"
	@echo "🔎 Checking for legacy trade_execution imports..."
	@if grep -rn "^from trade_execution\|^import trade_execution" --include="*.py" . ; then \
	  echo "❌ Found legacy 'trade_execution' imports. Please migrate to 'from ai_trading import ExecutionEngine'."; \
	  exit 1; \
	else \
	  echo "✅ No legacy 'trade_execution' imports found."; \
	fi

# Nightly or pre-release
test-e2e: clean install-dev validate-env
	$(PYTEST) -q -m "e2e"

test-ci: clean install-dev validate-env
	$(PYTEST) --maxfail=5 --disable-warnings --tb=short

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov coverage .coverage

coverage: install-dev
	PYTHONPATH=. pytest --cov=. --cov-report=html --cov-report=term

benchmark: install-dev
	PYTHONPATH=. pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=latest

# Linting and code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

mypy-check:
	mypy . --ignore-missing-imports

# Combined quality check
check: lint test-fast

run-backtest:
	$(PY) backtester.py \
	  --symbols AAPL MSFT GOOG AMZN TSLA \
	  --data-dir data \
	  --start 2023-01-01 \
	  --end 2024-01-01 \
	  --commission 0.005 \
	  --slippage-pips 0.1 \
	  --latency-bars 1

.PHONY: install install-dev test-all test-fast test-ci clean coverage benchmark lint mypy-check check run-backtest
