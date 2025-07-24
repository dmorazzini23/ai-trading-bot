#!/usr/bin/env bash
set -euo pipefail
echo "🔁 Starting AI Trading Bot Scheduler..."

cd /home/aiuser/ai-trading-bot
git config --global --add safe.directory /home/aiuser/ai-trading-bot

if [ -f .env ]; then
  echo "📦 Loading environment variables from .env"
  set +u; set -a
  source .env
  set +a; set -u
fi

if [ ! -d "venv" ]; then
  echo "🛠 Creating virtual environment and installing dependencies..."
  python3.12 -m venv venv
  venv/bin/pip install --upgrade pip setuptools wheel
  venv/bin/pip install -r requirements.txt
fi

source venv/bin/activate
export PYTHONUNBUFFERED=1
export WEB_CONCURRENCY=${WEB_CONCURRENCY:-1}

echo "🔍 Validating environment variables..."
python validate_env.py

echo "🚀 Starting core trading bot..."
exec python -u -m ai_trading.main --serve-api
