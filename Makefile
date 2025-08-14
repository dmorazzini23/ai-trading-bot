.PHONY: init test lint verify test-all

init:
	python -m pip install --upgrade pip
	@if [ -f requirements.txt ]; then \
		echo "[init] Installing base requirements.txt"; \
		pip install --retries 3 --timeout 60 -r requirements.txt || \
		( echo "[init] retrying base requirements"; pip install --retries 3 --timeout 60 -r requirements.txt ); \
	fi
	@if [ -f requirements-dev.txt ]; then \
		echo "[init] Installing requirements-dev.txt"; \
		pip install --retries 3 --timeout 60 -r requirements-dev.txt || \
		( echo "[init] retrying dev requirements"; pip install --retries 3 --timeout 60 -r requirements-dev.txt ); \
	fi
	# Minimal fallbacks (idempotent) to guarantee test discovery imports succeed
	pip install --retries 3 --timeout 60 pytest pytest-xdist || true

test-all: test  # AI-AGENT-REF: alias full test run

test:
	pytest -q -n auto --maxfail=1 --disable-warnings

lint:
	python -m py_compile $(shell git ls-files *.py)

verify:
	@if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi
	chmod +x scripts/quick_verify.sh
	./scripts/quick_verify.sh
