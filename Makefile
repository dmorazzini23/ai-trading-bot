benchmark:
	pytest tests/test_benchmarks.py --benchmark-only

profile:
	python profile_indicators.py

backtest:
	python backtest.py

gridsearch:
	python backtester/grid_runner.py
