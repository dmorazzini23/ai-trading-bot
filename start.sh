#!/usr/bin/env bash
set -euo pipefail

echo "🔁 Starting AI Trading Bot..."
source ~/.bashrc
cd ~/ai-trading-bot
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip setuptools >/dev/null
pip install --quiet -r requirements.txt
gunicorn -w 2 -b 0.0.0.0:${WEBHOOK_PORT:-9000} server:app &
python3 bot.py
