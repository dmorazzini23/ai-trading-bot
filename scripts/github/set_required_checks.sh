#!/usr/bin/env bash
set -euo pipefail

# Configure required status checks for the default protected branch.
# If branch protection does not exist yet, this script creates a minimal
# protection rule and applies the required checks.
# Usage:
#   scripts/github/set_required_checks.sh
#   BRANCH=main REQUIRED_CHECKS="CI / ci,CodeQL / Analyze (python),Workflow Lint / actionlint" \
#     scripts/github/set_required_checks.sh

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI is required."
  exit 1
fi

if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "ERROR: gh is not authenticated. Run: gh auth login"
  exit 1
fi

origin_url="$(git remote get-url origin)"
case "${origin_url}" in
  git@github.com:*)
    repo="${origin_url#git@github.com:}"
    repo="${repo%.git}"
    ;;
  https://github.com/*)
    repo="${origin_url#https://github.com/}"
    repo="${repo%.git}"
    ;;
  *)
    echo "ERROR: unable to infer owner/repo from origin URL: ${origin_url}"
    exit 1
    ;;
esac

branch="${BRANCH:-main}"
checks_csv="${REQUIRED_CHECKS:-CI / ci,CodeQL / Analyze (python),Workflow Lint / actionlint}"
IFS=',' read -r -a checks <<<"${checks_csv}"

declare -a cleaned_checks=()
for check in "${checks[@]}"; do
  trimmed="$(echo "${check}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  if [[ -n "${trimmed}" ]]; then
    cleaned_checks+=("${trimmed}")
  fi
done

if [[ "${#cleaned_checks[@]}" -eq 0 ]]; then
  echo "ERROR: REQUIRED_CHECKS resolved to an empty list."
  exit 1
fi

status_endpoint="repos/${repo}/branches/${branch}/protection/required_status_checks"
protection_endpoint="repos/${repo}/branches/${branch}/protection"

if gh api "${protection_endpoint}" >/dev/null 2>&1; then
  cmd=(
    gh api
    -X PATCH
    "${status_endpoint}"
    -f strict=true
  )
  for check in "${cleaned_checks[@]}"; do
    cmd+=(-F "contexts[]=${check}")
  done
  echo "INFO: updating required checks for existing protection rule on ${repo}@${branch}"
  "${cmd[@]}"
  echo "OK: required status checks updated."
else
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required to bootstrap branch protection."
    exit 1
  fi
  contexts_json="$(printf '%s\n' "${cleaned_checks[@]}" | jq -Rsc 'split("\n") | map(select(length > 0))')"
  payload="$(jq -n \
    --argjson contexts "${contexts_json}" \
    '{
      required_status_checks: {
        strict: true,
        contexts: $contexts
      },
      enforce_admins: false,
      required_pull_request_reviews: null,
      restrictions: null
    }')"
  echo "INFO: no branch protection rule found; creating minimal rule on ${repo}@${branch}"
  gh api -X PUT "${protection_endpoint}" --input - <<<"${payload}"
  echo "OK: branch protection created and required status checks configured."
fi
