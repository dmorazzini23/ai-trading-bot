#!/usr/bin/env bash
# AI-AGENT-REF: run safe lint auto-fixes
set -euo pipefail
python tools/codemods/fix_none_comparisons.py
ruff check --select F401,F841,E711,E402,F632 --fix .
ruff check --select F401,F841,E711,E402,F632 .
