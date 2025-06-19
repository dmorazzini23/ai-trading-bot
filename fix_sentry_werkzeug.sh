#!/bin/bash
set -e

echo "==> Activating virtual environment"
source venv/bin/activate

echo "==> Uninstalling current sentry-sdk"
pip uninstall -y sentry-sdk

echo "==> Installing sentry-sdk version 2.18.0 (compatible with Flask integration)"
pip install sentry-sdk==2.18.0

echo "==> Verifying FlaskIntegration import from sentry-sdk"
python -c "
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
print('FlaskIntegration import succeeded')
"

echo "==> Script completed successfully."

