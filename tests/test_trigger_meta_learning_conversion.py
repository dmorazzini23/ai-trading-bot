import sys
import tempfile
import os

# Mock the config module to avoid environment variable requirements
# Create a simple class-based approach that avoids singleton complexity

class TradingConfig:
    """Mock TradingConfig class for testing."""
    # Risk Management Parameters
    max_drawdown_threshold = 0.15
    daily_loss_limit = 0.03
    dollar_risk_limit = 0.05
    max_portfolio_risk = 0.025
    max_correlation_exposure = 0.15
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
    
    @classmethod
    def from_env(cls, mode="balanced"):
        return cls()
    
    def get_legacy_params(self):
        """Return legacy parameters for backward compatibility."""
        return {
            'conf_threshold': self.conf_threshold,
            'buy_threshold': self.buy_threshold,
            'min_confidence': self.min_confidence,
            'confirmation_count': self.confirmation_count,
            'take_profit_factor': self.take_profit_factor,
            'trailing_factor': self.trailing_factor,
            'scaling_factor': self.scaling_factor,
        }

class MockConfig:
    """Simple mock config that supports both attribute and import access."""
    # Class attributes that can be modified by tests
    TRADE_LOG_FILE = 'logs/trades.csv'
    VERBOSE_LOGGING = True
    SCHEDULER_SLEEP_SECONDS = 30.0
    NEWS_API_KEY = "test_news_api_key"
    TESTING = True
    REQUIRED_ENV_VARS = []
    SEED = 42
    ALPACA_DATA_FEED = "iex"
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    RATE_LIMIT_BUDGET = 190
    
    # Set TradingConfig as a class attribute for imports
    TradingConfig = TradingConfig
    
    # SGD Parameters from config.py
    SGD_PARAMS = {
        "loss": "squared_error",
        "learning_rate": "constant", 
        "eta0": 0.01,
        "penalty": "l2",
        "alpha": 0.0001,
        "random_state": 42,
        "max_iter": 1000,
        "tol": 1e-3
    }
    
    @classmethod
    def __getattr__(cls, name):
        """Return a default value for any missing attribute."""
        # Common default values for config attributes
        defaults = {
            'ALPACA_API_KEY': 'PKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'ALPACA_SECRET_KEY': 'SKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD',
            'BOT_MODE': 'balanced',
            'MODEL_PATH': 'trained_model.pkl',
            'WEBHOOK_SECRET': 'test-webhook-secret',
            'FLASK_PORT': 9000,
            'DISABLE_DAILY_RETRAIN': False,
            'SHADOW_MODE': False,
            'DRY_RUN': False,
        }
        
        if name in defaults:
            return defaults[name]
        
        # For any other attribute, return a reasonable default based on the name
        if name.endswith('_LIMIT') or name.endswith('_THRESHOLD'):
            return 0.05
        elif name.endswith('_PORT'):
            return 9000
        elif name.endswith('_BUDGET') or name.endswith('_SIZE'):
            return 100
        elif name.endswith('_MODE') or name.endswith('_FLAG'):
            return False
        elif name.endswith('_PATH') or name.endswith('_FILE'):
            return f"test_{name.lower()}"
        else:
            return None
    
    @staticmethod
    def reload_env():
        """Mock reload_env method."""
        pass
    
    @staticmethod
    def validate_env_vars():
        """Mock validate_env_vars method."""
        pass
    
    @staticmethod
    def validate_alpaca_credentials():
        """Mock validate_alpaca_credentials method."""
        pass
    
    @staticmethod
    def log_config(env_vars):
        """Mock log_config method."""
        pass
    
    @staticmethod
    def mask_secret(value: str, show_last: int = 4) -> str:
        """Return value with all but the last show_last characters masked."""
        if value is None:
            return ""
        return "*" * max(0, len(value) - show_last) + value[-show_last:]
    
    @staticmethod
    def get_env(key: str, default=None, required=False, reload=False):
        """Mock get_env method."""
        import os
        if reload:
            MockConfig.reload_env()
        
        # Mock common environment variables used in tests
        defaults = {
            "MODELS_DIR": "models",
            "MODEL_PATH": "trained_model.pkl",
            "ALPACA_API_KEY": "PKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "ALPACA_SECRET_KEY": "SKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD",
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
            "BOT_MODE": "balanced",
        }
        
        value = os.getenv(key, defaults.get(key, default))
        if required and not value:
            raise RuntimeError(f"Required environment variable {key} not set")
        return value

# Replace the config module with our mock
sys.modules['config'] = MockConfig

from ai_trading import meta_learning

def test_trigger_meta_learning_conversion_pure_meta_format():
    """Test trigger function with pure meta-learning format - should return True immediately."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write meta-learning format data
        f.write("symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n")
        f.write("TEST,2025-08-05T23:17:35Z,100.0,2025-08-05T23:18:35Z,105.0,10,buy,test_strategy,test,signal1+signal2,0.8,5.0\n")
        f.write("AAPL,2025-08-05T23:19:35Z,150.0,2025-08-05T23:20:35Z,155.0,5,buy,test_strategy,test,signal3,0.7,25.0\n")
        f.write("MSFT,2025-08-05T23:21:35Z,300.0,2025-08-05T23:22:35Z,295.0,2,sell,test_strategy,test,signal4,0.6,-10.0\n")
        test_file = f.name
    
    try:
        # Set the trade log file path
        MockConfig.TRADE_LOG_FILE = test_file
        
        # Test trade data
        test_trade = {
            'symbol': 'TEST', 
            'qty': 10, 
            'side': 'buy', 
            'price': 100.0, 
            'timestamp': '2025-08-05T23:17:35Z', 
            'order_id': 'test-001', 
            'status': 'filled'
        }
        
        # Verify quality report shows pure meta format
        quality_report = meta_learning.validate_trade_data_quality(test_file)
        assert quality_report['mixed_format_detected'] is False
        assert quality_report['audit_format_rows'] == 0
        assert quality_report['meta_format_rows'] > 0
        
        # Test the trigger function - should return True immediately (no conversion needed)
        result = meta_learning.trigger_meta_learning_conversion(test_trade)
        assert result is True
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_trigger_meta_learning_conversion_pure_audit_format():
    """Test trigger function with pure audit format - should attempt conversion."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write audit format data
        f.write("order_id,timestamp,symbol,side,qty,price,mode,status\n")
        f.write("123e4567-e89b-12d3-a456-426614174000,2025-08-05T23:17:35Z,TEST,buy,10,100.0,live,filled\n")
        f.write("234e5678-e89b-12d3-a456-426614174001,2025-08-05T23:18:35Z,TEST,sell,10,105.0,live,filled\n")
        f.write("345e6789-e89b-12d3-a456-426614174002,2025-08-05T23:19:35Z,AAPL,buy,5,150.0,live,filled\n")
        test_file = f.name
    
    try:
        # Set the trade log file path
        MockConfig.TRADE_LOG_FILE = test_file
        
        # Test trade data
        test_trade = {
            'symbol': 'TEST', 
            'qty': 10, 
            'side': 'buy', 
            'price': 100.0, 
            'timestamp': '2025-08-05T23:17:35Z', 
            'order_id': 'test-001', 
            'status': 'filled'
        }
        
        # Verify quality report shows pure audit format
        quality_report = meta_learning.validate_trade_data_quality(test_file)
        assert quality_report['mixed_format_detected'] is False
        assert quality_report['audit_format_rows'] > 0
        assert quality_report['meta_format_rows'] == 0
        
        # Test the trigger function - should attempt conversion and return True if successful
        result = meta_learning.trigger_meta_learning_conversion(test_trade)
        assert result is True  # Should succeed in conversion
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_trigger_meta_learning_conversion_mixed_format():
    """Test trigger function with mixed format - should attempt conversion."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write mixed format data (meta headers with audit data)
        f.write("symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n")
        f.write("123e4567-e89b-12d3-a456-426614174000,2025-08-05T23:17:35Z,TEST,buy,10,100.0,live,filled\n")
        f.write("234e5678-e89b-12d3-a456-426614174001,2025-08-05T23:18:35Z,TEST,sell,10,105.0,live,filled\n")
        test_file = f.name
    
    try:
        # Set the trade log file path
        MockConfig.TRADE_LOG_FILE = test_file
        
        # Test trade data
        test_trade = {
            'symbol': 'TEST', 
            'qty': 10, 
            'side': 'buy', 
            'price': 100.0, 
            'timestamp': '2025-08-05T23:17:35Z', 
            'order_id': 'test-001', 
            'status': 'filled'
        }
        
        # Verify quality report shows mixed format
        quality_report = meta_learning.validate_trade_data_quality(test_file)
        assert quality_report['mixed_format_detected'] is True
        
        # Test the trigger function - should attempt conversion and return True if successful
        result = meta_learning.trigger_meta_learning_conversion(test_trade)
        assert result is True  # Should succeed in conversion
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


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


def test_trigger_meta_learning_conversion_problem_statement_exact():
    """Test the exact scenario from the problem statement."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create exactly the scenario: mixed_format_detected=False, audit_format_rows=0, meta_format_rows=4
        f.write("symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n")
        f.write("TEST,2025-08-05T23:17:35Z,100.0,2025-08-05T23:18:35Z,105.0,10,buy,test_strategy,test,signal1+signal2,0.8,5.0\n")
        f.write("AAPL,2025-08-05T23:19:35Z,150.0,2025-08-05T23:20:35Z,155.0,5,buy,test_strategy,test,signal3,0.7,25.0\n")
        f.write("MSFT,2025-08-05T23:21:35Z,300.0,2025-08-05T23:22:35Z,295.0,2,sell,test_strategy,test,signal4,0.6,-10.0\n")
        f.write("GOOGL,2025-08-05T23:23:35Z,2500.0,2025-08-05T23:24:35Z,2505.0,1,buy,test_strategy,test,signal5,0.9,5.0\n")
        test_file = f.name
    
    try:
        MockConfig.TRADE_LOG_FILE = test_file
        
        test_trade = {
            'symbol': 'TEST', 
            'qty': 10, 
            'side': 'buy', 
            'price': 100.0, 
            'timestamp': '2025-08-05T23:17:35Z', 
            'order_id': 'test-001', 
            'status': 'filled'
        }
        
        # Verify we have the exact scenario from problem statement
        quality_report = meta_learning.validate_trade_data_quality(test_file)
        assert quality_report['mixed_format_detected'] is False
        assert quality_report['audit_format_rows'] == 0
        assert quality_report['meta_format_rows'] > 0  # Should be 5 (4 data + 1 header)
        
        # This should return True immediately (no conversion needed)
        result = meta_learning.trigger_meta_learning_conversion(test_trade)
        assert result is True, "Should return True for properly formatted meta-learning files"
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)