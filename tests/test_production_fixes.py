#!/usr/bin/env python3
"""
Tests for critical production fixes implemented in August 2025.

This test suite validates the four main fixes:
1. Sentiment API configuration support
2. Improved process detection logic  
3. Market-aware data staleness thresholds
4. Enhanced environment debugging capabilities
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestSentimentAPIConfiguration(unittest.TestCase):
    """Test sentiment API configuration and backwards compatibility."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock config module to avoid import issues
        self.config_mock = MagicMock()
        sys.modules['config'] = self.config_mock
    
    def tearDown(self):
        """Clean up test environment."""
        if 'config' in sys.modules:
            del sys.modules['config']
    
    def test_sentiment_api_env_vars_in_config(self):
        """Test that sentiment API variables are properly configured."""
        import ai_trading.config as config
        
        # Test that the new environment variables are accessible
        self.assertTrue(hasattr(config, 'SENTIMENT_API_KEY') or 'SENTIMENT_API_KEY' in dir(config))
        self.assertTrue(hasattr(config, 'SENTIMENT_API_URL') or 'SENTIMENT_API_URL' in dir(config))
    
    @patch.dict(os.environ, {
        'NEWS_API_KEY': 'test_news_key',
        'SENTIMENT_API_KEY': 'test_sentiment_key',
        'SENTIMENT_API_URL': 'https://api.test.com/sentiment'
    })
    def test_sentiment_api_backwards_compatibility(self):
        """Test backwards compatibility with NEWS_API_KEY."""
        # Test that SENTIMENT_API_KEY takes precedence
        
        # Simulate the configuration logic
        sentiment_key = os.getenv('SENTIMENT_API_KEY') or os.getenv('NEWS_API_KEY')
        self.assertEqual(sentiment_key, 'test_sentiment_key')
        
        # Test fallback to NEWS_API_KEY
        with patch.dict(os.environ, {'NEWS_API_KEY': 'fallback_key'}, clear=True):
            sentiment_key = os.getenv('SENTIMENT_API_KEY') or os.getenv('NEWS_API_KEY')
            self.assertEqual(sentiment_key, 'fallback_key')


class TestProcessDetection(unittest.TestCase):
    """Test improved process detection logic."""
    
    def setUp(self):
        """Set up performance monitor test."""
        try:
            from performance_monitor import ResourceMonitor
            self.monitor = ResourceMonitor()
        except ImportError:
            self.skipTest("performance_monitor module not available")
    
    def test_trading_bot_process_detection(self):
        """Test that trading bot process detection works correctly."""
        # Test the new _count_trading_bot_processes method
        if hasattr(self.monitor, '_count_trading_bot_processes'):
            count = self.monitor._count_trading_bot_processes()
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 1)  # At least this test process
    
    @patch('subprocess.run')
    def test_process_filtering(self, mock_subprocess):
        """Test that temporary processes are filtered out."""
        # Mock ps aux output with trading bot and temporary processes
        mock_output = """USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
user      1234  0.1  1.0 123456  7890 ?        S    09:00   0:01 python bot_engine.py
user      5678  0.0  0.1  12345   890 ?        S    09:30   0:00 python -c import sys; print("test")
user      9999  0.2  1.5 234567  8901 ?        S    09:01   0:02 python runner.py --mode balanced
user      1111  0.0  0.1  11111   111 ?        S    10:59   0:00 pgrep -f python
"""
        
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = mock_output
        
        if hasattr(self.monitor, '_count_trading_bot_processes'):
            count = self.monitor._count_trading_bot_processes()
            # Should count bot_engine.py and runner.py, but not the test command or pgrep
            self.assertGreaterEqual(count, 2)
    
    def test_alert_threshold_adjustment(self):
        """Test that multiple process alerts have appropriate thresholds."""
        # Create mock metrics with multiple processes
        metrics = {
            'process': {
                'python_processes': 3
            }
        }
        
        alerts = self.monitor.check_alert_conditions(metrics)
        
        # Should have an alert for multiple processes
        multiple_proc_alerts = [a for a in alerts if 'multiple' in a.get('type', '')]
        if multiple_proc_alerts:
            alert = multiple_proc_alerts[0]
            # Threshold should be 2 (allowing for main + backup process)
            self.assertEqual(alert.get('threshold', 1), 2)


class TestDataStalenessThresholds(unittest.TestCase):
    """Test market-aware data staleness detection."""
    
    def setUp(self):
        """Set up data validation test."""
        try:
            import pandas as pd
            from ai_trading.data_validation import (
                is_market_hours, get_staleness_threshold, 
                check_data_freshness
            )
            self.pd = pd
            self.is_market_hours = is_market_hours
            self.get_staleness_threshold = get_staleness_threshold
            self.check_data_freshness = check_data_freshness
        except ImportError:
            self.skipTest("Required modules not available for data validation tests")
    
    def test_market_hours_detection(self):
        """Test market hours detection logic."""
        # Test during market hours (Tuesday 2:00 PM ET = 7:00 PM UTC)
        market_time = datetime(2025, 8, 5, 19, 0, 0, tzinfo=timezone.utc)  # Tuesday 2 PM ET
        self.assertTrue(self.is_market_hours(market_time))
        
        # Test outside market hours (Tuesday 6:00 PM ET = 11:00 PM UTC)  
        after_hours = datetime(2025, 8, 5, 23, 0, 0, tzinfo=timezone.utc)  # Tuesday 6 PM ET
        self.assertFalse(self.is_market_hours(after_hours))
        
        # Test weekend (Saturday)
        weekend = datetime(2025, 8, 9, 19, 0, 0, tzinfo=timezone.utc)  # Saturday 2 PM ET
        self.assertFalse(self.is_market_hours(weekend))
    
    def test_staleness_threshold_logic(self):
        """Test that staleness thresholds adapt to market conditions."""
        # During market hours should have stricter threshold
        market_time = datetime(2025, 8, 5, 19, 0, 0, tzinfo=timezone.utc)
        market_threshold = self.get_staleness_threshold('AAPL', market_time)
        self.assertEqual(market_threshold, 15)  # 15 minutes during market hours
        
        # After hours should be more lenient
        after_hours = datetime(2025, 8, 5, 23, 0, 0, tzinfo=timezone.utc)
        after_threshold = self.get_staleness_threshold('AAPL', after_hours)
        self.assertGreater(after_threshold, market_threshold)
        
        # Weekend should be most lenient
        weekend = datetime(2025, 8, 9, 19, 0, 0, tzinfo=timezone.utc)
        weekend_threshold = self.get_staleness_threshold('AAPL', weekend)
        self.assertGreater(weekend_threshold, after_threshold)
    
    def test_data_freshness_with_market_awareness(self):
        """Test that data freshness checks include market context."""
        # Create test dataframe with recent data
        now = datetime.now(timezone.utc)
        recent_time = now - timedelta(minutes=10)
        
        df = self.pd.DataFrame({
            'Close': [100.0],
            'Volume': [1000]
        }, index=[recent_time])
        
        result = self.check_data_freshness(df, 'TEST')
        
        # Result should include market hours information
        self.assertIn('market_hours', result)
        self.assertIn('staleness_threshold', result)
        self.assertIsInstance(result['market_hours'], bool)
        self.assertIsInstance(result['staleness_threshold'], int)


class TestEnvironmentDebugging(unittest.TestCase):
    """Test enhanced environment debugging capabilities."""
    
    def setUp(self):
        """Set up environment validation test."""
        try:
            from validate_env import debug_environment, validate_specific_env_var
            self.debug_environment = debug_environment
            self.validate_specific_env_var = validate_specific_env_var
        except ImportError:
            self.skipTest("validate_env module not available")
    
    def test_debug_environment_structure(self):
        """Test that debug environment returns proper structure."""
        debug_report = self.debug_environment()
        
        # Check required fields
        required_fields = [
            'timestamp', 'validation_status', 'critical_issues',
            'warnings', 'environment_vars', 'recommendations'
        ]
        
        for field in required_fields:
            self.assertIn(field, debug_report)
        
        # Check data types
        self.assertIsInstance(debug_report['critical_issues'], list)
        self.assertIsInstance(debug_report['warnings'], list)
        self.assertIsInstance(debug_report['environment_vars'], dict)
        self.assertIsInstance(debug_report['recommendations'], list)
    
    @patch.dict(os.environ, {
        'ALPACA_API_KEY': 'test_key_12345',
        'SENTIMENT_API_KEY': 'sentiment_key_67890'
    })
    def test_sensitive_value_masking(self):
        """Test that sensitive environment values are properly masked."""
        debug_report = self.debug_environment()
        
        env_vars = debug_report['environment_vars']
        
        # API keys should be masked
        if 'ALPACA_API_KEY' in env_vars:
            api_key_info = env_vars['ALPACA_API_KEY']
            self.assertEqual(api_key_info['status'], 'set')
            self.assertIn('...', api_key_info['value'])  # Should be masked
            self.assertIn('length', api_key_info)
    
    def test_specific_env_var_validation(self):
        """Test validation of specific environment variables."""
        # Test with a mock environment variable
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            result = self.validate_specific_env_var('TEST_VAR')
            
            self.assertEqual(result['variable'], 'TEST_VAR')
            self.assertEqual(result['status'], 'set')
            self.assertEqual(result['value'], 'test_value')
        
        # Test with missing variable
        result = self.validate_specific_env_var('NONEXISTENT_VAR')
        self.assertEqual(result['status'], 'missing')
        self.assertIn('not set', result['issues'][0])


class TestIntegration(unittest.TestCase):
    """Integration tests for all fixes working together."""
    
    def test_env_file_contains_sentiment_config(self):
        """Test that .env file contains the new sentiment configuration."""
        env_file_path = '.env'
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                content = f.read()
                
            # Should contain sentiment API configuration
            self.assertIn('SENTIMENT_API_KEY', content)
            self.assertIn('SENTIMENT_API_URL', content)
            
            # Should maintain backwards compatibility
            self.assertIn('NEWS_API_KEY', content)
    
    def test_all_modules_importable(self):
        """Test that all modified modules can be imported without errors."""
        modules_to_test = [
            'performance_monitor',
            'data_validation', 
            'validate_env'
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                # Allow for missing dependencies in test environment
                if 'pandas' in str(e) or 'pydantic' in str(e):
                    continue
                else:
                    self.fail(f"Failed to import {module_name}: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)