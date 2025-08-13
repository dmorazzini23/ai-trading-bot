.PHONY: init test lint verify

init:
	python -m pip install --upgrade pip
	@if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
	# common fallbacks used in CI/dev
	pip install pytest pytest-xdist pandas yfinance alpaca-trade-api pandas-ta pydantic

test:
	pytest -q -n auto --maxfail=1 --disable-warnings

lint:
	python -m py_compile $(shell git ls-files '*.py')

verify:
        @if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
        chmod +x scripts/quick_verify.sh
        ./scripts/quick_verify.sh
