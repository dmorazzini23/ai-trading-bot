#!/usr/bin/env python3
"""
Environment configuration helper for test execution.
This script sets environment variables to handle missing dependencies.
"""

import os
import sys

def setup_test_environment():
    """Set environment variables required for testing."""
    
    # Set testing flags first to enable test mode
    os.environ["PYTEST_RUNNING"] = "1"
    os.environ["TESTING"] = "1"
    
    # Required environment variables for basic import
    test_env_vars = {
        "ALPACA_API_KEY": "PKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "ALPACA_SECRET_KEY": "SKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD", 
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "WEBHOOK_SECRET": "test-webhook-secret-for-testing",
        "FLASK_PORT": "9000",
        "BOT_MODE": "balanced",
        "DOLLAR_RISK_LIMIT": "0.02",
        "BUY_THRESHOLD": "0.5",
        "TRADE_LOG_FILE": "test_trades.csv",
        "SEED": "42",
        "RATE_LIMIT_BUDGET": "190",
        "REBALANCE_INTERVAL_MIN": "60",
        "SHADOW_MODE": "true",
        "DISABLE_DAILY_RETRAIN": "true",
        "DRY_RUN": "true",
        "MEMORY_OPTIMIZED": "1",
        "FINNHUB_API_KEY": "test_finnhub_key",
        "NEWS_API_KEY": "test_news_key"
    }
    
    # Set environment variables
    for key, value in test_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    print("✅ Test environment variables configured")

def create_minimal_dependencies():
    """Create minimal stub modules for missing dependencies."""
    
    # Create stub for missing modules if needed
    stubs = {}
    
    # Check if python-dotenv is available
    try:
        import dotenv
        print("✅ python-dotenv is available")
    except ImportError:
        print("ℹ️ python-dotenv not available, using config fallback")
    
    # Check if pydantic is available  
    try:
        import pydantic
        import pydantic_settings
        print("✅ pydantic and pydantic-settings are available")
    except ImportError:
        print("ℹ️ pydantic-settings not available, using config fallback")
    
    return stubs

def main():
    """Main configuration function."""
    print("🔧 Setting up test environment...")
    
    setup_test_environment()
    create_minimal_dependencies()
    
    print("✅ Test environment setup completed!")
    
    # Test basic imports
    try:
        print("🧪 Testing basic imports...")
        
        # Test config import
        sys.path.insert(0, "/home/runner/work/ai-trading-bot/ai-trading-bot")
        import config
        print("✅ config module imported successfully")
        
        # Test that environment variables are accessible
        print(f"✅ ALPACA_API_KEY: {os.getenv('ALPACA_API_KEY')[:10]}...")
        print(f"✅ Testing mode: {config.TESTING}")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)