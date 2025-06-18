#!/usr/bin/env bash
set -eo pipefail

echo "🔁 Starting AI Trading Bot..."

cd /root/ai-trading-bot

# Load environment variables from .env if present
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Create virtualenv if missing
if [ ! -d venv ]; then
  python3.12 -m venv venv
fi

# Activate virtualenv
source venv/bin/activate

# Force-reinstall setuptools, upgrade pip, then install requirements (show errors)
pip install --upgrade --force-reinstall setuptools pip wheel || exit 1
pip install -r requirements.txt || exit 1
gunicorn -w 2 -b 0.0.0.0:${WEBHOOK_PORT:-9000} server:app &
python3.12 bot.py
