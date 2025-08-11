#!/usr/bin/env bash
set -euo pipefail
pat='try\s*:\s*$\n\s*(from|import)\s+.*\n\s*except\s+ImportError'
if git grep -nE "$pat" -- 'ai_trading/**/*.py' | grep -q .; then
  echo "ERROR: ImportError guards detected in production code."
  git grep -nE "$pat" -- 'ai_trading/**/*.py'
  exit 1
fi