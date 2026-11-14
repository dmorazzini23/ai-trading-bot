import os
import sys
import tempfile
import types
from typing import Any, cast

# Mock the config module to avoid environment variable requirements
# Create a simple class-based approach that avoids singleton complexity

class TradingConfig:
    """Mock TradingConfig class for testing."""
    # Risk Management Parameters
    max_drawdown_threshold = 0.15
    daily_loss_limit = 0.03
    dollar_risk_limit = 0.05
    max_portfolio_risk = 0.025
    sector_exposure_cap = 0.15
    max_sector_concentration = 0.15
    min_liquidity_threshold = 1000000
    position_size_min_usd = 100.0
    max_position_size = 8000
    max_position_size_pct = 0.25

    # Kelly Criterion Parameters
    kelly_fraction = 0.6
    kelly_fraction_max = 0.25
    min_sample_size = 20
    confidence_level = 0.90
    lookback_periods = 252
    rebalance_frequency = 21

    # Trading Mode Parameters
    conf_threshold = 0.75
    buy_threshold = 0.1
    min_confidence = 0.6
    confirmation_count = 2
    take_profit_factor = 1.8
    trailing_factor = 1.2
    scaling_factor = 0.3
    TRADE_LOG_FILE = "trades.csv"

    @classmethod
    def from_env(cls, mode="balanced"):
        return cls()

# Replace the config module with our mock
config_stub = types.ModuleType("config")
cast(Any, config_stub).TradingConfig = TradingConfig
cast(Any, config_stub).TRADE_LOG_FILE = TradingConfig.TRADE_LOG_FILE
MockConfig: Any = config_stub
sys.modules["config"] = config_stub

from ai_trading import meta_learning








def test_trigger_meta_learning_conversion_missing_file():
    """Test trigger function with missing file - should return False."""
    # Set a non-existent file path
    MockConfig.TRADE_LOG_FILE = '/tmp/non_existent_file.csv'

    test_trade = {
        'symbol': 'TEST',
        'qty': 10,
        'side': 'buy',
        'price': 100.0,
        'timestamp': '2025-08-05T23:17:35Z',
        'order_id': 'test-001',
        'status': 'filled'
    }

    # Test the trigger function - should return False for missing file
    result = meta_learning.trigger_meta_learning_conversion(test_trade)
    assert result is False


