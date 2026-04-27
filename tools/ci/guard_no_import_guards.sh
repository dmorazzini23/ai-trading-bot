#!/usr/bin/env bash
set -euo pipefail

diff_args=()
if [[ -n "${BASE_REF:-}" ]] && git rev-parse --verify "${BASE_REF}" >/dev/null 2>&1; then
  diff_args=("${BASE_REF}...HEAD")
elif [[ -n "${GITHUB_BASE_REF:-}" ]] && git rev-parse --verify "origin/${GITHUB_BASE_REF}" >/dev/null 2>&1; then
  diff_args=("origin/${GITHUB_BASE_REF}...HEAD")
elif git rev-parse --verify origin/main >/dev/null 2>&1; then
  diff_args=("origin/main...HEAD")
fi

diff_output=""
if ((${#diff_args[@]} > 0)); then
  diff_output="$(git diff -U0 "${diff_args[@]}" -- 'ai_trading/**/*.py' || true)"
fi
diff_output+=$'\n'
diff_output+="$(git diff -U0 --cached -- 'ai_trading/**/*.py' || true)"
diff_output+=$'\n'
diff_output+="$(git diff -U0 -- 'ai_trading/**/*.py' || true)"

violations="$(
  printf '%s\n' "${diff_output}" \
    | awk '
      /^diff --git / { file=$4; sub(/^b\//, "", file); next }
      /^\+\+\+ / { file=$2; sub(/^b\//, "", file); next }
      /^\+[^+]/ && $0 ~ /except[[:space:]]+ImportError/ {
        print file ":" substr($0, 2)
      }
    '
)"

if [[ -n "${violations}" ]]; then
  echo "ERROR: new ImportError guards detected in production code." >&2
  printf '%s\n' "${violations}" >&2
  exit 1
fi
