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

# Kill any process using the Flask port before launching Gunicorn
if lsof -ti tcp:"${FLASK_PORT:-9000}" >/dev/null; then
  echo "🔪 Port ${FLASK_PORT:-9000} in use, terminating existing processes..."
  lsof -ti tcp:"${FLASK_PORT:-9000}" | xargs -r kill -TERM || true
  sleep 2
  lsof -ti tcp:"${FLASK_PORT:-9000}" | xargs -r kill -KILL 2>/dev/null || true
fi

echo "🌐 Launching Gunicorn server..."

exec gunicorn -w 4 -b 0.0.0.0:${FLASK_PORT:-9000} \
  --access-logfile - \
  --error-logfile - \
  --capture-output \
  --enable-stdio-inheritance \
  server:app

# Run the trading bot as the main foreground process
exec ./venv/bin/python -u bot.py

# Launch Gunicorn with explicit port variable
echo "🌐 Launching Gunicorn server on port $FLASK_PORT..."
exec ./venv/bin/gunicorn -w 4 -b 0.0.0.0:$FLASK_PORT \
  --access-logfile - --error-logfile - server:app
