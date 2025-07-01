test-all: clean
	pytest --maxfail=3 --disable-warnings -n auto -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov coverage

coverage:
	pytest --cov=. --cov-report=html

benchmark:
	pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=latest
