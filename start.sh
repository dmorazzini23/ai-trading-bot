#!/usr/bin/env bash
set -euo pipefail

echo "🔁 Starting AI Trading Bot..."

cd /home/aiuser/ai-trading-bot

# Load environment variables from .env if present
if [ -f .env ]; then
  set +u
  set -a
  source .env
  set +a
  set -u
fi

# Provide a default port if FLASK_PORT is not set
export FLASK_PORT=${FLASK_PORT:-9000}

# Validate env (optional, your script)
echo "🔍 Validating environment variables..."
python validate_env.py

# Activate virtualenv
if [ ! -d venv ]; then
  echo "🛠 Creating virtualenv and installing dependencies..."
  python3.12 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# Launch Gunicorn with explicit port variable
echo "🌐 Launching Gunicorn server on port $FLASK_PORT..."
exec ./venv/bin/gunicorn -w 4 -b 0.0.0.0:$FLASK_PORT \
  --access-logfile - --error-logfile - server:app
