# Install dependencies
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

# Environment validation
validate-env:
	python scripts/validate_test_environment.py

# Testing targets
test-all: clean install-dev validate-env
	PYTHONPATH=. pytest --maxfail=3 --disable-warnings -n auto -v

test-fast: clean install-dev validate-env
	PYTHONPATH=. pytest --maxfail=1 --disable-warnings -x

test-ci: clean install-dev validate-env
	PYTHONPATH=. pytest --maxfail=5 --disable-warnings --tb=short

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
	python backtester.py \
	  --symbols AAPL MSFT GOOG AMZN TSLA \
	  --data-dir data \
	  --start 2023-01-01 \
	  --end 2024-01-01 \
	  --commission 0.005 \
	  --slippage-pips 0.1 \
	  --latency-bars 1

.PHONY: install install-dev test-all test-fast test-ci clean coverage benchmark lint mypy-check check run-backtest
