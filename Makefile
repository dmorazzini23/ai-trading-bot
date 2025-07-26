test-all: clean
	PYTHONPATH=. pytest --maxfail=3 --disable-warnings -n auto -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov coverage

coverage:
	pytest --cov=. --cov-report=html

benchmark:
	pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=latest
run-backtest:
	python backtester.py \
	  --symbols AAPL MSFT GOOG AMZN TSLA \
	  --data-dir data/historical \
	  --start 2020-01-01 \
	  --end 2021-01-01 \
	  --commission 0.005 \
	  --slippage-pips 0.1 \
	  --latency-bars 1
	
