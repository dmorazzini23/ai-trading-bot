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

# Now require WEBHOOK_SECRET
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

# Launch HTTP server if available, passing through WEBHOOK_SECRET
if command -v gunicorn >/dev/null; then
  exec gunicorn \
    --workers 2 \
    --bind 0.0.0.0:${WEBHOOK_PORT:-9000} \
    --env WEBHOOK_SECRET="$WEBHOOK_SECRET" \
    server:app &
else
  echo "⚠️ gunicorn not found; skipping server"
fi

# Finally, run the bot using venv’s Python
exec python -u bot.py
