benchmark:
	pytest tests/test_benchmarks.py --benchmark-only

profile:
	python profile_indicators.py

backtest:
	python backtest.py

gridsearch:
	python backtester/grid_runner.py

test-backtester:
	pytest tests/test_backtest_smoke.py
	pytest tests/test_integration.py
	pytest tests/test_risk_engine_module.py
	python -m backtester.plot
