#!/usr/bin/env bash
set -euo pipefail

echo "🔁 Starting AI Trading Bot Scheduler..."

cd /home/aiuser/ai-trading-bot

# Avoid git "dubious ownership" warnings when the repo is mounted by root
git config --global --add safe.directory /home/aiuser/ai-trading-bot

# Load environment variables from .env if present
if [ -f .env ]; then
  set +u
  set -a
  source .env
  set +a
  set -u
fi

# Ensure virtualenv exists and install dependencies if needed
if [ ! -d venv ]; then
  echo "🛠 Creating virtualenv and installing dependencies..."
  python3.12 -m venv venv
  venv/bin/pip install --upgrade pip setuptools wheel
  venv/bin/pip install -r requirements.txt
fi

# Activate the virtualenv
source venv/bin/activate
export WEB_CONCURRENCY=${WEB_CONCURRENCY:-1}

echo "🔍 Validating environment variables..."
python validate_env.py

echo "🤖 Launching scheduler loop..."
exec python -u bot_engine.py
