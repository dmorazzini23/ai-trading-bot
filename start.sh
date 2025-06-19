#!/usr/bin/env bash
set -euo pipefail

echo "🔁 Starting AI Trading Bot..."

cd /root/ai-trading-bot

# Load environment variables from .env if present
if [ -f .env ]; then
  set +u
  set -a
  source .env
  set +a
  set -u
fi

# Require WEBHOOK_SECRET environment variable
export WEBHOOK_SECRET=${WEBHOOK_SECRET:?ERROR: WEBHOOK_SECRET must be set in .env}

# Run environment validation script to catch missing vars early
echo "🔍 Validating environment variables..."
python validate_env.py

# Ensure Python 3.12 venv exists and activate it
if [ ! -d venv ]; then
  echo "🛠 Creating new virtualenv and installing dependencies..."
  python3.12 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# Optionally launch the Flask server in the background if server.py exists
if [ -f server.py ]; then
  echo "🌐 Launching Flask server..."
  python -u server.py &
else
  echo "⚠️ server.py not found; skipping HTTP server"
fi

# Run the trading bot as the main foreground process
exec python -u bot.py
