#!/usr/bin/env bash
set -euo pipefail

# AI-AGENT-REF: install runtime and dev dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt

# AI-AGENT-REF: lint and run tests
ruff --force-exclude .
pytest -n auto --disable-warnings -q

