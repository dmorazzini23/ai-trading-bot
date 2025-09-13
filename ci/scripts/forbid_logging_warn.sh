#!/usr/bin/env bash
set -euo pipefail

if git ls-files '*.py' | xargs grep -n "logging\\.warn(" >/dev/null; then
  echo 'ERROR: logging.warn present' >&2
  exit 1
fi

echo 'OK'
