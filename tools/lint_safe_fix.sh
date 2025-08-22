#!/usr/bin/env bash
set -euo pipefail

# 1) Pass 1: imports + trivial formatting
ruff check . --select E,W,F401,I,RUF100 --fix || true

# 2) Pass 2: re-run to clean newly-exposed issues
ruff check . --select E,W,F401,I,RUF100 --fix || true

# 3) Optional targeted codemod for None comparisons when Ruff left stragglers
python tools/codemods/fix_none_comparisons.py || true

# 4) Final sweep (don’t fail build; we only want the report)
ruff check . --output-format=full --exit-zero | tee artifacts/ruff-after.txt
