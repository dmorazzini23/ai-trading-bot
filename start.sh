#!/usr/bin/env bash
set -euo pipefail
cd /home/aiuser/ai-trading-bot
source venv/bin/activate
python validate_env.py
exec python -u -m ai_trading.main
