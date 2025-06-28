benchmark:
	pytest tests/test_benchmarks.py --benchmark-only -n auto

profile:
	pytest --profile-svg --maxfail=1 --disable-warnings

test-all-backtests:
	pytest -n auto tests/test_equity_curve.py
	pytest -n auto tests/test_slippage.py
	pytest -n auto tests/test_hyperparams.py
	pytest -n auto tests/test_regime_filters.py
	pytest -n auto tests/test_parallel_speed.py
	pytest -n auto tests/test_grid_sanity.py
