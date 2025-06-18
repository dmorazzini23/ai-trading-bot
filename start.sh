#!/usr/bin/env bash
set -eo pipefail

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

# Now require WEBHOOK_SECRET (placeholder or real)
export WEBHOOK_SECRET=${WEBHOOK_SECRET:?ERROR: WEBHOOK_SECRET must be set in .env}

# Ensure Python 3.12 venv exists
if [ ! -d venv ]; then
  echo "🛠 Creating new virtualenv and installing dependencies..."
  python3.12 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# If you still need the HTTP endpoints, run the Flask server directly
if [ -f server.py ]; then
  echo "🌐 Launching Flask server..."
  python -u server.py &
else
  echo "⚠️ server.py not found; skipping HTTP server"
fi

# Finally, run the trading bot
exec python -u bot.py
