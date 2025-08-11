#!/usr/bin/env bash
set -euo pipefail
git grep -nE '\bexec\s*\(' -- 'ai_trading/**/*.py' | (! grep .)
git grep -nE '(^|[^.])\beval\s*\(' -- 'ai_trading/**/*.py' | (! grep .)