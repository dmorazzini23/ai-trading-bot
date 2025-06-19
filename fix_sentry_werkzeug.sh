#!/bin/bash
set -e

echo "==> Activating virtual environment"
source venv/bin/activate

echo "==> Uninstalling current sentry-sdk"
pip uninstall -y sentry-sdk

echo "==> Installing sentry-sdk version 2.18.0 (compatible with Flask integration)"
pip install sentry-sdk==2.18.0

echo "==> Verifying sentry-sdk version and Flask integration import"
python -c "
import sentry_sdk
print('sentry-sdk version:', sentry_sdk.__version__)
from sentry_sdk.integrations.flask import FlaskIntegration
print('FlaskIntegration import succeeded')
"

echo "==> Restarting your AI trading scheduler service"
sudo systemctl restart ai-trading-scheduler.service

echo "==> Tail logs to verify service startup"
sudo journalctl -u ai-trading-scheduler.service -f
