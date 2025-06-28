coverage:
	pytest --cov=. --cov-report=html

benchmark:
	pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=latest

profile:
	python profile_indicators.py

test-all:
	pytest -n auto --maxfail=3 --disable-warnings

test-all-backtests:
	pytest tests/test_equity_curve.py
	pytest tests/test_slippage.py
	pytest tests/test_hyperparams.py
	pytest tests/test_regime_filters.py
	pytest tests/test_parallel_speed.py
	pytest tests/test_grid_sanity.py
