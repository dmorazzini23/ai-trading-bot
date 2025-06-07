#!/usr/bin/env bash
set -euo pipefail

echo "🔁 Starting AI Trading Bot..."
source ~/.bashrc
cd ~/AI_TRADING_BOT
source venv/bin/activate
gunicorn -w 2 -b 0.0.0.0:${WEBHOOK_PORT:-9000} server:app &
python3 bot.py
