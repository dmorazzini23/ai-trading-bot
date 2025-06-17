#!/usr/bin/env bash
set -eo pipefail

echo "🔁 Starting AI Trading Bot..."

# Change to project directory
cd /root/ai-trading-bot

# Load environment variables from .env if present
if [ -f .env ]; then
  set +u
  set -a
  source .env
  set +a
  set -u
fi

# Ensure Python 3.12 binary exists
PYTHON_BIN=$(command -v python3.12 || echo "/usr/bin/python3.12")

# Create virtualenv if missing
if [ ! -d venv ]; then
  "$PYTHON_BIN" -m venv venv
fi

# Activate virtualenv
source venv/bin/activate

# Force-reinstall setuptools/pip/wheel, upgrade pip, then install requirements
pip install --upgrade --force-reinstall setuptools pip wheel || exit 1
pip install -r requirements.txt            || exit 1

# Launch HTTP server if available
if command -v gunicorn >/dev/null; then
  gunicorn -w2 -b0.0.0.0:${WEBHOOK_PORT:-9000} server:app &
else
  echo "⚠️ gunicorn not found; skipping server"
fi

# Finally, run the bot
exec "$PYTHON_BIN" -u bot.py
