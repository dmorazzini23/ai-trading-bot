#!/usr/bin/env python3
import logging

"""
API Key Configuration Verification Script

This script helps verify that your API key configuration is set up correctly
and provides guidance on any issues found.
"""

import os
import sys
from pathlib import Path

from ai_trading.config import management as config
from ai_trading.config.management import TradingConfig

CONFIG = TradingConfig()


def check_env_file():
    """Check if .env file exists and has proper format."""
    env_path = Path('.env')

    if not env_path.exists():
        return False, "❌ .env file not found. Run: cp .env.example .env"

    try:
        with open(env_path) as f:
            content = f.read()

        # Check for required variables
        required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL']
        missing_vars = []
        placeholder_vars = []

        # Load environment first to check if keys are set via env vars
        from dotenv import load_dotenv
        load_dotenv('.env', override=True)

        for var in required_vars:
            env_value = os.getenv(var)
            if var not in content:
                missing_vars.append(var)
            elif env_value and env_value.startswith('YOUR_'):
                placeholder_vars.append(var)

        if missing_vars:
            return False, f"❌ Missing variables in .env: {', '.join(missing_vars)}"

        # Only warn about placeholders if not overridden by environment variables
        if placeholder_vars:
            return False, f"⚠️  Placeholder values detected in .env: {', '.join(placeholder_vars)}\n   Please replace YOUR_* placeholders with your real API keys"

        return True, "✅ .env file format looks good"

    except ImportError:
        return False, "❌ python-dotenv not installed. Run: pip install python-dotenv"
    # noqa: BLE001 TODO: narrow exception
    except Exception as e:
        return False, f"❌ Error reading .env file: {e}"


def check_api_keys():
    """Check if API keys are properly configured."""
    try:
        # Try to load environment
        from dotenv import load_dotenv
        load_dotenv('.env', override=True)

        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL')

        issues = []

        # Check API key
        if not api_key:
            issues.append("ALPACA_API_KEY not set")
        elif api_key.startswith('YOUR_'):
            issues.append("ALPACA_API_KEY still contains placeholder text")
        elif len(api_key) < 20:
            issues.append("ALPACA_API_KEY appears too short")

        # Check secret key
        if not secret_key:
            issues.append("ALPACA_SECRET_KEY not set")
        elif secret_key.startswith('YOUR_'):
            issues.append("ALPACA_SECRET_KEY still contains placeholder text")
        elif len(secret_key) < 30:
            issues.append("ALPACA_SECRET_KEY appears too short")

        # Check base URL
        if not base_url:
            issues.append("ALPACA_BASE_URL not set")
        elif not base_url.startswith('https://'):
            issues.append("ALPACA_BASE_URL should start with https://")

        if issues:
            return False, "❌ API key issues found:\n   " + "\n   ".join(issues)

        # Determine if using paper trading
        is_paper = 'paper' in base_url.lower()
        env_type = "Paper Trading (Safe)" if is_paper else "Live Trading (Real Money!)"

        return True, f"✅ API keys configured correctly\n   Environment: {env_type}"

    except ImportError:
        return False, "❌ python-dotenv not installed. Run: pip install python-dotenv"
    # noqa: BLE001 TODO: narrow exception
    except Exception as e:
        return False, f"❌ Error checking API keys: {e}"


def check_config_import():
    """Check if the config module can be imported successfully."""
    try:
        # Set test environment to avoid validation errors
        os.environ['TESTING'] = '1'

        # Check if keys are accessible
        has_api_key = bool(config.ALPACA_API_KEY and config.ALPACA_API_KEY != 'YOUR_ALPACA_API_KEY_HERE')
        has_secret = bool(config.ALPACA_SECRET_KEY and config.ALPACA_SECRET_KEY != 'YOUR_ALPACA_SECRET_KEY_HERE')

        if has_api_key and has_secret:
            return True, "✅ Configuration module imports successfully with API keys"
        else:
            return False, "⚠️  Configuration imports but API keys not properly set"

    # noqa: BLE001 TODO: narrow exception
    except Exception as e:
        return False, f"❌ Error importing config: {e}"


def print_setup_instructions():
    """Print setup instructions."""
    logging.info("""
🔧 Setup Instructions:

1. Get your API keys:
   → Visit: https://app.alpaca.markets/paper/dashboard/overview
   → Generate API keys (use Paper Trading for testing))

2. Configure your .env file:
   → Copy: cp .env.example .env
   → Edit .env and replace YOUR_* placeholders with real keys

3. Verify your setup:
   → Run this script again: python verify_config.py

📖 For detailed instructions, see: docs/API_KEY_SETUP.md
""")


def main():
    """Main verification function."""
    logging.info("🔍 AI Trading Bot - API Key Configuration Verification\n")

    all_good = True

    # Check .env file
    env_ok, env_msg = check_env_file()
    logging.info(str(env_msg))
    if not env_ok:
        all_good = False

    # Check API keys
    api_ok, api_msg = check_api_keys()
    logging.info(str(api_msg))
    if not api_ok:
        all_good = False

    # Check config import
    config_ok, config_msg = check_config_import()
    logging.info(str(config_msg))
    if not config_ok:
        all_good = False

    logging.info(str("\n" + "="*60))

    if all_good:
        logging.info("🎉 SUCCESS: Your API key configuration is ready!")
        logging.info("\nNext steps:")
        logging.info("  → Run the bot: python -m ai_trading")
        logging.info("  → Or run tests: python -m pytest")
    else:
        logging.info("❌ ISSUES FOUND: Please fix the above issues before running the bot.")
        print_setup_instructions()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
