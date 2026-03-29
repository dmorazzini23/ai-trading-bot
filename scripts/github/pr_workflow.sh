#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required"
  exit 1
fi

REPO="${1:-}"
PR="${2:-}"

if [[ -z "${REPO}" || -z "${PR}" ]]; then
  echo "usage: $0 <owner/repo> <pr-number>"
  exit 2
fi

echo "=== PR metadata ==="
gh pr view "${PR}" --repo "${REPO}" --json number,title,state,isDraft,mergeStateStatus,headRefName,baseRefName | jq .

echo "=== Changed files ==="
gh pr view "${PR}" --repo "${REPO}" --json files | jq '.files[] | {path:.path, additions:.additions, deletions:.deletions}'

echo "=== Review threads (flat comments) ==="
gh api "repos/${REPO}/pulls/${PR}/comments?per_page=100" | jq 'map({id,path,line,author:.user.login,body})'

echo "=== PR conversation comments ==="
gh api "repos/${REPO}/issues/${PR}/comments?per_page=100" | jq 'map({id,author:.user.login,body,created_at})'

echo "=== Checks summary ==="
head_sha="$(gh pr view "${PR}" --repo "${REPO}" --json headRefOid -q .headRefOid)"
gh api "repos/${REPO}/commits/${head_sha}/status" | jq '{state,total_count,statuses:[.statuses[]|{context,state,description,target_url}]}'

echo "PR_WORKFLOW_REPORT_OK"

