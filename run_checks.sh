#!/usr/bin/env bash

set -e

echo "Installing main requirements..."
pip install -r requirements.txt

echo "Installing test requirements..."
pip install -r requirements-test.txt

# Ensure linting tools are available
pip install flake8 isort==5.12.0 pylint pytest-cov >/dev/null

echo "Running flake8..."
flake8

echo "Checking import order with isort..."
isort --check-only .

echo "Running pylint..."
# Run pylint on all Python files excluding tests to reduce noise
pylint $(git ls-files '*.py' | grep -v '^tests/' | tr '\n' ' ')

echo "Running pytest with coverage..."
pytest --cov=.
