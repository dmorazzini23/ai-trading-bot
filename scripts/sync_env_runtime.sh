#!/usr/bin/env bash
set -euo pipefail
cd /home/aiuser/ai-trading-bot

python3 scripts/runtime_env_sync.py --src .env --dst .env.runtime
