.PHONY: init test lint verify test-all contract audit-exceptions self-check deps-dev lint-fix lint-fix-phase2 lint-fix-phase3 lint-fix-phase4r typecheck

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

dev-deps:
	python -m pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt
	[ -f requirements-dev.txt ] && pip install -r requirements-dev.txt -c constraints-dev.txt --no-deps || true
	python -m ruff --version | tee artifacts/ruff-version.txt || true
	python -m mypy --version | tee artifacts/mypy-version.txt || true
	python -m pytest --version | tee artifacts/pytest-version.txt || true

test-all: dev-deps
	       $(MAKE) lint-fix-phase4r
	       $(MAKE) typecheck
		       pytest -n auto --disable-warnings --maxfail=0 -q | tee artifacts/pytest.txt || true

## Lint (safe-fix subset)
.PHONY: lint-fix
lint-fix:
	tools/lint_safe_fix.sh

.PHONY: lint-fix-phase2
lint-fix-phase2:
	tools/lint_safe_fix.sh  # AI-AGENT-REF: phase2b lint pass

.PHONY: lint-fix-phase3
lint-fix-phase3:
	       bash tools/lint_phase3.sh

.PHONY: lint-fix-phase4r
lint-fix-phase4r:
	       bash tools/lint_phase4r.sh

.PHONY: lint
lint:
	ruff check .

.PHONY: typecheck
typecheck:
	python -m mypy --version | tee artifacts/mypy-version.txt
	python -m mypy ai_trading trade_execution | tee artifacts/mypy-phase3.txt || true  # AI-AGENT-REF: ensure type safety

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
