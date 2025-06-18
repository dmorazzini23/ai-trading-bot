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

# Launch HTTP server if available
if command -v gunicorn >/dev/null; then
  gunicorn -w2 -b0.0.0.0:${WEBHOOK_PORT:-9000} server:app &
else
  echo "⚠️ gunicorn not found; skipping server"
fi

# Finally, run the bot using venv’s Python
exec python -u bot.py
