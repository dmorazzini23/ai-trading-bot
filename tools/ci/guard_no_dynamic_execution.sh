#!/usr/bin/env bash
set -euo pipefail
if git grep -nE '\bexec\s*\(' -- 'ai_trading/**/*.py' | grep -q .; then
  echo "ERROR: exec() detected in production code."
  git grep -nE '\bexec\s*\(' -- 'ai_trading/**/*.py'
  exit 1
fi
if git grep -nE '(^|[^.])\beval\s*\(' -- 'ai_trading/**/*.py' | grep -q .; then
  echo "ERROR: raw eval() detected in production code."
  git grep -nE '(^|[^.])\beval\s*\(' -- 'ai_trading/**/*.py'
  exit 1
fi