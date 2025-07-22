#!/usr/bin/env bash
set -euo pipefail

echo "🔁 Starting AI Trading Bot Scheduler..."

cd /home/aiuser/ai-trading-bot

# Ensure Git doesn't throw ownership warnings in cloud-mounted volumes
git config --global --add safe.directory /home/aiuser/ai-trading-bot

# Load environment variables from .env (if not already injected by systemd)
if [ -f .env ]; then
  echo "📦 Loading environment variables from .env"
  set +u
  set -a
  source .env
  set +a
  set -u
fi

# Create virtual environment if missing
if [ ! -d "venv" ]; then
  echo "🛠 Creating virtual environment and installing dependencies..."
  python3.12 -m venv venv
  venv/bin/pip install --upgrade pip setuptools wheel
  venv/bin/pip install -r requirements.txt
fi

# Activate virtual environment
source venv/bin/activate

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1
export WEB_CONCURRENCY=${WEB_CONCURRENCY:-1}

echo "🔍 Validating environment variables..."
python validate_env.py

echo "🚀 Starting core trading bot..."
python -m ai_trading
