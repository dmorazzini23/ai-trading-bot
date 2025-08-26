import logging
import os
import sys
from pathlib import Path
from ai_trading.settings import _secret_to_str, get_settings
from ai_trading.env import ensure_dotenv_loaded
from ai_trading.config.management import get_env, reload_env

def check_env_file():
    """Check if .env file exists and has proper format."""
    env_path = Path('.env')
    if not env_path.exists():
        return (False, '❌ .env file not found. Run: cp .env.example .env')
    try:
        with open(env_path) as f:
            content = f.read()
        required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL']
        missing_vars = []
        sample_vars = []
        loaded = reload_env(str(env_path), override=True)
        if loaded is None:
            return (False, '❌ python-dotenv not installed. Run: pip install python-dotenv')
        for var in required_vars:
            env_value = get_env(var)
            if var not in content:
                missing_vars.append(var)
            elif env_value and str(env_value).startswith('YOUR_'):
                sample_vars.append(var)
        if missing_vars:
            return (False, f"❌ Missing variables in .env: {', '.join(missing_vars)}")
        if sample_vars:
            return (False, f"⚠️  Sample values detected in .env: {', '.join(sample_vars)}\n   Please replace YOUR_* sample values with your real API keys")
        return (True, '✅ .env file format looks good')
    except (OSError, PermissionError, KeyError, ValueError, TypeError) as e:
        return (False, f'❌ Error reading .env file: {e}')

def check_api_keys():
    """Check if API keys are properly configured."""
    try:
        loaded = reload_env('.env', override=True)
        if loaded is None:
            return (False, '❌ python-dotenv not installed. Run: pip install python-dotenv')
        api_key = get_env('ALPACA_API_KEY')
        secret_key = get_env('ALPACA_SECRET_KEY')
        base_url = get_env('ALPACA_BASE_URL')
        issues = []
        if not api_key:
            issues.append('ALPACA_API_KEY not set')
        elif api_key.startswith('YOUR_'):
            issues.append('ALPACA_API_KEY still contains sample text')
        elif len(api_key) < 20:
            issues.append('ALPACA_API_KEY appears too short')
        if not secret_key:
            issues.append('ALPACA_SECRET_KEY not set')
        elif secret_key.startswith('YOUR_'):
            issues.append('ALPACA_SECRET_KEY still contains sample text')
        elif len(secret_key) < 30:
            issues.append('ALPACA_SECRET_KEY appears too short')
        if not base_url:
            issues.append('ALPACA_BASE_URL not set')
        elif not base_url.startswith('https://'):
            issues.append('ALPACA_BASE_URL should start with https://')
        if issues:
            return (False, '❌ API key issues found:\n   ' + '\n   '.join(issues))
        is_paper = 'paper' in base_url.lower()
        env_type = 'Paper Trading (Safe)' if is_paper else 'Live Trading (Real Money!)'
        return (True, f'✅ API keys configured correctly\n   Environment: {env_type}')
    except (KeyError, ValueError, TypeError) as e:
        return (False, f'❌ Error checking API keys: {e}')

def check_config_import():
    """Check if settings can be loaded and contain API keys."""
    try:
        os.environ['TESTING'] = '1'
        s = get_settings()
        has_api_key = bool(s.alpaca_api_key and s.alpaca_api_key != 'YOUR_ALPACA_API_KEY_HERE')
        secret = _secret_to_str(getattr(s, 'alpaca_secret_key', None))
        has_secret = bool(secret and secret != 'YOUR_ALPACA_SECRET_KEY_HERE')
        if has_api_key and has_secret:
            return (True, '✅ Settings load successfully with API keys')
        return (False, '⚠️ Settings loaded but API keys not properly set')
    except (KeyError, ValueError, TypeError) as e:
        return (False, f'❌ Error loading settings: {e}')

def print_setup_instructions():
    """Print setup instructions."""
    logging.info('\n🔧 Setup Instructions:\n\n1. Get your API keys:\n   → Visit: https://app.alpaca.markets/paper/dashboard/overview\n   → Generate API keys (use Paper Trading for testing))\n\n2. Configure your .env file:\n   → Copy: cp .env.example .env\n   → Edit .env and replace YOUR_* sample values with real keys\n\n3. Verify your setup:\n   → Run this script again: python verify_config.py\n\n📖 For detailed instructions, see: docs/API_KEY_SETUP.md\n')

def main():
    """Main verification function."""
    logging.info('🔍 AI Trading Bot - API Key Configuration Verification\n')
    ensure_dotenv_loaded()
    all_good = True
    env_ok, env_msg = check_env_file()
    logging.info(str(env_msg))
    if not env_ok:
        all_good = False
    api_ok, api_msg = check_api_keys()
    logging.info(str(api_msg))
    if not api_ok:
        all_good = False
    config_ok, config_msg = check_config_import()
    logging.info(str(config_msg))
    if not config_ok:
        all_good = False
    logging.info(str('\n' + '=' * 60))
    if all_good:
        logging.info('🎉 SUCCESS: Your API key configuration is ready!')
        logging.info('\nNext steps:')
        logging.info('  → Run the bot: python -m ai_trading')
        logging.info('  → Or run tests: python -m pytest')
    else:
        logging.info('❌ ISSUES FOUND: Please fix the above issues before running the bot.')
        print_setup_instructions()
        return 1
    return 0
if __name__ == '__main__':
    sys.exit(main())