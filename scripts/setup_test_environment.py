#!/usr/bin/env python3
"""
Comprehensive environment setup script for AI Trading Bot test suite.
Handles network connectivity issues, dependency installation, and environment configuration.
"""

import os
import sys
import subprocess
import time
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Network timeout settings for pip
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 5

def setup_test_environment():
    """Setup comprehensive test environment with network access and dependencies."""
    
    # Create test environment file
    env_file_path = create_test_environment_file()
    
    # Install dependencies with retry logic
    install_dependencies_with_retry()
    
    # Validate test environment 
    validate_environment()
    
    logger.info("✅ Test environment setup completed successfully!")
    return env_file_path

def create_test_environment_file() -> str:
    """Create .env.test file with all required environment variables for testing."""
    
    test_env_content = """# Test Environment Configuration
# This file provides all required environment variables for test execution

# Alpaca API Configuration (using test/paper trading format)
ALPACA_API_KEY=PKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ
ALPACA_SECRET_KEY=SKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_FEED=iex

# API Keys (test values)
FINNHUB_API_KEY=test_finnhub_key
NEWS_API_KEY=test_news_key
FUNDAMENTAL_API_KEY=test_fundamental_key
IEX_API_TOKEN=test_iex_token

# Flask and Webhook Configuration
FLASK_PORT=9000
WEBHOOK_SECRET=test-webhook-secret-for-testing
WEBHOOK_PORT=9000

# Trading Bot Configuration
BOT_MODE=balanced
MODEL_PATH=trained_model.pkl
HALT_FLAG_PATH=halt.flag
MAX_PORTFOLIO_POSITIONS=20
LIMIT_ORDER_SLIPPAGE=0.005
BUY_THRESHOLD=0.5
SLIPPAGE_THRESHOLD=0.003

# Health Check Configuration
HEALTHCHECK_PORT=8081
RUN_HEALTHCHECK=0
MIN_HEALTH_ROWS=30
MIN_HEALTH_ROWS_DAILY=5

# Trading Parameters
FORCE_TRADES=0
REBALANCE_INTERVAL_MIN=1440
SHADOW_MODE=true
DRY_RUN=true
DISABLE_DAILY_RETRAIN=true
TRADE_LOG_FILE=test_trades.csv

# Risk Management
DISASTER_DD_LIMIT=0.2
SECTOR_EXPOSURE_CAP=0.4
MAX_OPEN_POSITIONS=10
WEEKLY_DRAWDOWN_LIMIT=0.15
VOLUME_THRESHOLD=50000
DOLLAR_RISK_LIMIT=0.02
EQUITY_EXPOSURE_CAP=2.5
PORTFOLIO_EXPOSURE_CAP=2.5

# Model Paths
MODEL_RF_PATH=model_rf.pkl
MODEL_XGB_PATH=model_xgb.pkl
MODEL_LGB_PATH=model_lgb.pkl
RL_MODEL_PATH=rl_agent.zip
USE_RL_AGENT=false

# Rate Limiting
FINNHUB_RPM=60
MINUTE_CACHE_TTL=60
RATE_LIMIT_BUDGET=190

# Testing Flags
TESTING=1
PYTEST_RUNNING=1
MEMORY_OPTIMIZED=1
SCHEDULER_SLEEP_SECONDS=60
SEED=42

# Environment markers for specific test behavior
MAX_PORTFOLIO_POSITIONS=10
"""
    
    env_test_path = "/home/runner/work/ai-trading-bot/ai-trading-bot/.env.test"
    
    with open(env_test_path, 'w') as f:
        f.write(test_env_content)
    
    logger.info(f"✅ Created test environment file: {env_test_path}")
    return env_test_path

def install_package_with_retry(package: str, max_retries: int = MAX_RETRIES) -> bool:
    """Install a package with retry logic for network issues."""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Installing {package} (attempt {attempt + 1}/{max_retries})")
            
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--timeout", str(DEFAULT_TIMEOUT),
                "--retries", "3",
                "--user",
                package
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT * 2
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully installed {package}")
                return True
            else:
                logger.warning(f"❌ Failed to install {package}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⏱️ Timeout installing {package} (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"❌ Error installing {package}: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    
    logger.error(f"❌ Failed to install {package} after {max_retries} attempts")
    return False

def install_dependencies_with_retry():
    """Install required dependencies with retry logic and batch processing."""
    
    # Critical packages for testing
    critical_packages = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0", 
        "pytest-xdist>=3.0.0",
        "pytest-asyncio>=0.20.2",
        "hypothesis>=6.0.0"
    ]
    
    # Core scientific packages
    core_packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "pydantic-settings>=0.1.0",
        "pydantic>=2.0"
    ]
    
    # Extended packages for full functionality
    extended_packages = [
        "pyarrow>=12.0.0",
        "scikit-learn>=1.4.2", 
        "joblib>=1.3.0",
        "requests>=2.31.0",
        "flask>=2.3.0",
        "tenacity==8.2.2"
    ]
    
    # Install packages in priority order
    package_groups = [
        ("Critical Test Packages", critical_packages),
        ("Core Scientific Packages", core_packages), 
        ("Extended Functionality", extended_packages)
    ]
    
    for group_name, packages in package_groups:
        logger.info(f"📦 Installing {group_name}...")
        
        successful = 0
        for package in packages:
            if install_package_with_retry(package):
                successful += 1
        
        logger.info(f"✅ Installed {successful}/{len(packages)} packages from {group_name}")
        
        # Don't fail completely if extended packages fail
        if group_name == "Critical Test Packages" and successful == 0:
            logger.error("❌ Failed to install critical test packages!")
            sys.exit(1)

def validate_environment():
    """Validate that the environment is properly configured for testing."""
    
    logger.info("🔍 Validating test environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error(f"❌ Python {python_version} is too old. Requires Python 3.8+")
        sys.exit(1)
    
    logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check critical packages
    critical_imports = [
        'pytest',
        'hypothesis', 
        'os',
        'sys',
        'pathlib',
        'subprocess'
    ]
    
    for package in critical_imports:
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError as e:
            logger.error(f"❌ {package} import failed: {e}")
    
    # Check environment file exists
    env_test_path = "/home/runner/work/ai-trading-bot/ai-trading-bot/.env.test"
    if os.path.exists(env_test_path):
        logger.info(f"✅ Test environment file exists: {env_test_path}")
    else:
        logger.error(f"❌ Test environment file missing: {env_test_path}")

def create_network_retry_wrapper():
    """Create a wrapper script for network operations with retry logic."""
    
    wrapper_content = '''#!/bin/bash
# Network operations wrapper with retry logic

MAX_RETRIES=3
RETRY_DELAY=5

retry_command() {
    local cmd="$1"
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        echo "Attempt $attempt/$MAX_RETRIES: $cmd"
        
        if eval "$cmd"; then
            echo "✅ Command succeeded on attempt $attempt"
            return 0
        else
            echo "❌ Command failed on attempt $attempt"
            if [ $attempt -lt $MAX_RETRIES ]; then
                echo "Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi
        
        ((attempt++))
    done
    
    echo "❌ Command failed after $MAX_RETRIES attempts"
    return 1
}

# Export function for use in other scripts
export -f retry_command
'''
    
    wrapper_path = "/home/runner/work/ai-trading-bot/ai-trading-bot/scripts/network_retry.sh"
    
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    os.chmod(wrapper_path, 0o755)
    logger.info(f"✅ Created network retry wrapper: {wrapper_path}")
    return wrapper_path

def main():
    """Main setup function."""
    logger.info("🚀 Starting comprehensive test environment setup...")
    
    # Create network retry wrapper
    create_network_retry_wrapper()
    
    # Setup test environment
    env_file = setup_test_environment()
    
    logger.info("🎉 Test environment setup completed!")
    logger.info(f"Environment file: {env_file}")
    logger.info("You can now run tests with: PYTHONPATH=. pytest --disable-warnings")

if __name__ == "__main__":
    main()