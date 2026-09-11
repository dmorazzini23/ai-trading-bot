#!/usr/bin/env bash
# Explicit task presets; no changes to permissions, hooks or trading runtime.
set -euo pipefail
mode=${1:-standard}
if (($#)); then shift; fi
case "$mode" in
  routine) model=gpt-5.6-luna; effort=low ;;
  standard) model=gpt-6-astra; effort=low ;;
  deep) model=gpt-6-astra; effort=medium ;;
  *) printf '%s\n' 'Usage: bash scripts/codex_task.sh {routine|standard|deep} [Codex arguments]' >&2; exit 2 ;;
esac
exec codex --model "$model" -c "model_reasoning_effort=\"$effort\"" -c tool_output_token_limit=2000 "$@"
