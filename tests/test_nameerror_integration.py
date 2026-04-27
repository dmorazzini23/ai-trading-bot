"""Integration test to verify the NameError fix for BUY_THRESHOLD"""
import os
import subprocess
import sys
from pathlib import Path
import pytest


pytestmark = pytest.mark.integration

def test_bot_engine_import_no_nameerror():
    """Test that bot_engine can be imported without NameError for BUY_THRESHOLD.
    
    This test creates a controlled environment and tries to import bot_engine,
    specifically checking for the NameError that was occurring before the fix.
    """

    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for deprecated_key in ("ALPACA_BASE_URL", "ALPACA_API_URL", "TRADING_MODE"):
        env.pop(deprecated_key, None)
    env.update(
        {
            "PYTHONPATH": str(project_root),
            "PYTEST_RUNNING": "1",
            "ALPACA_API_KEY": "FAKE_TEST_API_KEY_NOT_REAL_123456789",
            "ALPACA_SECRET_KEY": "FAKE_TEST_SECRET_KEY_NOT_REAL_123456789",
            "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
            "WEBHOOK_SECRET": "fake-test-webhook-not-real",
            "FLASK_PORT": "9000",
            "AI_TRADING_TRADING_MODE": "balanced",
            "DOLLAR_RISK_LIMIT": "0.05",
            "TESTING": "1",
            "TRADE_LOG_FILE": "test_trades.csv",
            "SEED": "42",
            "RATE_LIMIT_BUDGET": "190",
            "DISABLE_DAILY_RETRAIN": "True",
            "DRY_RUN": "True",
            "SHADOW_MODE": "True",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ai_trading.core import bot_engine; "
            "assert bot_engine.__name__ == 'ai_trading.core.bot_engine'",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    test_bot_engine_import_no_nameerror()
