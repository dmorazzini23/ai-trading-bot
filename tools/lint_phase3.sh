#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

# 1) Auto-fix safe rules
ruff --version | tee artifacts/ruff-version.txt
ruff check . --fix --exit-zero

# 2) Re-run to capture remaining issues (JSON + text)
ruff check . --exit-zero --output-format json > artifacts/ruff.json
ruff check . --exit-zero --statistics > artifacts/ruff.txt

# 3) Make a compact "top rules" histogram for quick triage
python tools/ruff_hist.py artifacts/ruff.json > artifacts/ruff-top-rules.tsv

# 4) Gate only for egregious regressions; don't fail CI yet
python - <<'PY'
import json, sys
data = json.load(open("artifacts/ruff.json"))
remaining = len(data)  # one entry per violation
print(f"Remaining Ruff violations: {remaining}")
# Soft target: ≤1500; do not exit nonzero yet, we’re iterating
PY
