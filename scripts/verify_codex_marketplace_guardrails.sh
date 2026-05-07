#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing required Codex guardrail path: $path" >&2
    exit 1
  fi
}

require_absent_pattern() {
  local pattern="$1"
  local path="$2"
  if rg -n "$pattern" "$path" >/dev/null; then
    echo "forbidden Codex guardrail pattern found: $pattern in $path" >&2
    exit 1
  fi
}

require_path "$repo_root/.codex/hooks.json"
require_path "$repo_root/.codex/hooks/aitmpl-codex--automation--agents-md-loader"
require_path "$repo_root/.codex/hooks/aitmpl-codex--security--secret-scanner"
require_path "$repo_root/.codex/hooks/aitmpl-codex--security--dangerous-command-blocker"
require_path "$repo_root/.codex/hooks/aitmpl-codex--development-tools--command-logger"
require_path "$repo_root/.codex/hooks/aitmpl-codex--automation--change-logger"

require_path "$repo_root/.codex/skills/bug-hunt-swarm/SKILL.md"
require_path "$repo_root/.codex/skills/review-swarm/SKILL.md"
require_path "$repo_root/.codex/skills/project-skill-audit/SKILL.md"
require_path "$repo_root/.codex/skills/orchestrate-batch-refactor/SKILL.md"
require_path "$repo_root/.codex/skills/context7-cli/SKILL.md"

require_path "$repo_root/.agents/plugins/marketplace.json"
require_path "$repo_root/plugins/sentry/.codex-plugin/plugin.json"

require_absent_pattern "auto-git-add|git-add-changes|smart-commit|auto-push" "$repo_root/.codex/hooks.json"
require_absent_pattern "slack-notifications|slack-detailed-notifications|slack-error-notifications" "$repo_root/.codex/hooks.json"
require_absent_pattern "format-python-files|smart-formatting|run-tests-after-changes" "$repo_root/.codex/hooks.json"

echo "codex marketplace guardrails verified"
