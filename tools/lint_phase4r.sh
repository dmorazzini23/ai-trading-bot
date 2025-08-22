#!/usr/bin/env bash
set -euo pipefail
python tools/codemod_none_is.py
ruff check . --fix --exit-zero
ruff check . --exit-zero | tee artifacts/ruff.txt
python tools/ruff_histogram.py < artifacts/ruff.txt | tee artifacts/ruff-top-rules.txt
