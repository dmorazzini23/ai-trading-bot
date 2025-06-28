test-all:
	pytest -n auto --maxfail=3 --disable-warnings

coverage:
	pytest --cov=. --cov-report=html

benchmark:
	pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=latest
