#!/usr/bin/env bash
set -e

# ——— Your droplet’s SSH info ———
SERVER="root@143.110.157.152"
APP_DIR="$HOME/ai-trading-bot"
BRANCH="main"

echo "Deploying branch '$BRANCH' to $SERVER:$APP_DIR …"

ssh "$SERVER" << EOF
  cd "$APP_DIR"
  git fetch origin "$BRANCH"
  git reset --hard "origin/$BRANCH"
  if [ ! -d venv ]; then
    python3 -m venv venv
  fi
  source venv/bin/activate
  pip install --upgrade pip setuptools >/dev/null
  pip install --quiet -r requirements.txt
  sudo systemctl restart tradingbot
  echo "✅ Bot restarted on $SERVER"
EOF

echo "🎉 Done."
