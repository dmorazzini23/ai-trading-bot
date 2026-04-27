#!/usr/bin/env bash
set -euo pipefail

failures=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && failures+=("${line}")
done < <(git grep -nE '\bexec\s*\(' -- 'ai_trading/**/*.py' || true)
while IFS= read -r line; do
  [[ -n "${line}" ]] && failures+=("${line}")
done < <(git grep -nE '(^|[^.])\beval\s*\(' -- 'ai_trading/**/*.py' || true)

if ((${#failures[@]} > 0)); then
  echo "ERROR: dynamic exec/eval detected in production code." >&2
  printf '%s\n' "${failures[@]}" >&2
  exit 1
fi
