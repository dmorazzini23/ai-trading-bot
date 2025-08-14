.PHONY: init test lint verify test-all

init:
	python -c "import sys; assert (3,12) <= sys.version_info < (3,13), f'Python {sys.version.split()[0]} detected; require >=3.12,<3.13'" || (echo 'WARNING: Non-3.12 Python detected; continuing for tooling only.' && true)
	python -m pip install --upgrade pip
	@if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
	# Ensure core tools are present even if prior step was interrupted
	python -m pip install "pytest" "pytest-xdist" "tzlocal>=5.2,<6"

test:
	pytest -q -n auto --maxfail=1 --disable-warnings

test-all: test

lint:
	python -m py_compile $(shell git ls-files '*.py')

verify:
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
	chmod +x scripts/quick_verify.sh
	./scripts/quick_verify.sh
