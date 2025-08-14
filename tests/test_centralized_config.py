#!/usr/bin/env python3
"""
Test suite for centralized trading configuration system.

This test validates that the centralized TradingConfig system works correctly
across all trading modes and parameter categories.
"""

import os
import pytest
os.environ["TESTING"] = "1"

from ai_trading.config import TradingConfig
from ai_trading.core.bot_engine import BotMode


class TestCentralizedConfig:
    """Test the centralized trading configuration system."""
    
    def test_trading_config_initialization(self):
        """Test that TradingConfig initializes correctly."""
        config = TradingConfig()
        
        # Test default values are set
        assert config.kelly_fraction == 0.6
        assert config.conf_threshold == 0.75
        assert config.daily_loss_limit == 0.03
        assert config.capital_cap == 0.25
        assert config.max_position_size == 8000
        
    def test_trading_config_from_env(self):
        """Test loading configuration from environment."""
        config = TradingConfig.from_env()
        
        # Should load successfully
        assert isinstance(config, TradingConfig)
        assert config.kelly_fraction is not None
        assert config.conf_threshold is not None
        
    def test_mode_specific_configurations(self):
        """Test that each mode has appropriate parameter values."""
        conservative = TradingConfig.from_env("conservative")
        balanced = TradingConfig.from_env("balanced")
        aggressive = TradingConfig.from_env("aggressive")
        
        # Conservative mode should have lower risk parameters
        assert conservative.kelly_fraction < balanced.kelly_fraction
        assert conservative.conf_threshold > balanced.conf_threshold
        assert conservative.daily_loss_limit < balanced.daily_loss_limit
        assert conservative.capital_cap < balanced.capital_cap
        assert conservative.confirmation_count > balanced.confirmation_count
        
        # Aggressive mode should have higher risk parameters
        assert aggressive.kelly_fraction > balanced.kelly_fraction
        assert aggressive.conf_threshold < balanced.conf_threshold
        assert aggressive.daily_loss_limit > balanced.daily_loss_limit
        assert aggressive.capital_cap > balanced.capital_cap
        assert aggressive.confirmation_count < balanced.confirmation_count
        
    def test_conservative_mode_parameters(self):
        """Test conservative mode specific values."""
        config = TradingConfig.from_env("conservative")
        
        assert config.kelly_fraction == 0.25
        assert config.conf_threshold == 0.85
        assert config.daily_loss_limit == 0.03
        assert config.capital_cap == 0.20
        assert config.confirmation_count == 3
        assert config.take_profit_factor == 1.5
        assert config.max_position_size == 5000
        
    def test_balanced_mode_parameters(self):
        """Test balanced mode specific values."""
        config = TradingConfig.from_env("balanced")
        
        assert config.kelly_fraction == 0.6
        assert config.conf_threshold == 0.75
        assert config.daily_loss_limit == 0.05  # Updated to reflect balanced mode
        assert config.capital_cap == 0.25
        assert config.confirmation_count == 2
        assert config.take_profit_factor == 1.8
        assert config.max_position_size == 8000
        
    def test_aggressive_mode_parameters(self):
        """Test aggressive mode specific values."""
        config = TradingConfig.from_env("aggressive")
        
        assert config.kelly_fraction == 0.75
        assert config.conf_threshold == 0.65
        assert config.daily_loss_limit == 0.08
        assert config.capital_cap == 0.30
        assert config.confirmation_count == 1
        assert config.take_profit_factor == 2.5
        assert config.max_position_size == 12000
        
    def test_legacy_parameter_interface(self):
        """Test that legacy parameter interface works correctly."""
        config = TradingConfig.from_env("balanced")
        legacy_params = config.get_legacy_params()
        
        # Check that legacy parameter names are available
        assert "KELLY_FRACTION" in legacy_params
        assert "CONF_THRESHOLD" in legacy_params
        assert "CONFIRMATION_COUNT" in legacy_params
        assert "TAKE_PROFIT_FACTOR" in legacy_params
        assert "DAILY_LOSS_LIMIT" in legacy_params
        assert "CAPITAL_CAP" in legacy_params
        assert "TRAILING_FACTOR" in legacy_params
        assert "BUY_THRESHOLD" in legacy_params
        
        # Check values match config
        assert legacy_params["KELLY_FRACTION"] == config.kelly_fraction
        assert legacy_params["CONF_THRESHOLD"] == config.conf_threshold
        assert legacy_params["CAPITAL_CAP"] == config.capital_cap
        
    def test_bot_mode_integration(self):
        """Test that BotMode class integrates correctly with centralized config."""
        conservative_mode = BotMode("conservative")
        balanced_mode = BotMode("balanced")
        aggressive_mode = BotMode("aggressive")
        
        # Test that each mode has the correct parameters
        cons_params = conservative_mode.get_config()
        bal_params = balanced_mode.get_config()
        agg_params = aggressive_mode.get_config()
        
        # Conservative mode checks
        assert cons_params["KELLY_FRACTION"] == 0.25
        assert cons_params["CONF_THRESHOLD"] == 0.85
        assert cons_params["CONFIRMATION_COUNT"] == 3
        
        # Balanced mode checks
        assert bal_params["KELLY_FRACTION"] == 0.6
        assert bal_params["CONF_THRESHOLD"] == 0.75
        assert bal_params["CONFIRMATION_COUNT"] == 2
        
        # Aggressive mode checks
        assert agg_params["KELLY_FRACTION"] == 0.75
        assert agg_params["CONF_THRESHOLD"] == 0.65
        assert agg_params["CONFIRMATION_COUNT"] == 1
        
    def test_parameter_completeness(self):
        """Test that all required parameters are present in the configuration."""
        config = TradingConfig.from_env("balanced")
        
        # Risk management parameters
        assert hasattr(config, 'max_drawdown_threshold')
        assert hasattr(config, 'daily_loss_limit')
        assert hasattr(config, 'dollar_risk_limit')
        assert hasattr(config, 'max_portfolio_risk')
        assert hasattr(config, 'max_position_size')
        assert hasattr(config, 'kelly_fraction')
        assert hasattr(config, 'capital_cap')
        
        # Trading mode parameters
        assert hasattr(config, 'conf_threshold')
        assert hasattr(config, 'buy_threshold')
        assert hasattr(config, 'confirmation_count')
        assert hasattr(config, 'take_profit_factor')
        assert hasattr(config, 'trailing_factor')
        
        # Signal processing parameters
        assert hasattr(config, 'signal_confirmation_bars')
        assert hasattr(config, 'signal_period')
        assert hasattr(config, 'fast_period')
        assert hasattr(config, 'slow_period')
        
        # Execution parameters
        assert hasattr(config, 'limit_order_slippage')
        assert hasattr(config, 'max_slippage_bps')
        assert hasattr(config, 'participation_rate')
        assert hasattr(config, 'pov_slice_pct')
        assert hasattr(config, 'order_timeout_seconds')
        
    def test_to_dict_conversion(self):
        """Test that configuration can be converted to dictionary."""
        config = TradingConfig.from_env("balanced")
        config_dict = config.to_dict()
        
        # Check that dictionary contains expected keys
        assert "kelly_fraction" in config_dict
        assert "conf_threshold" in config_dict
        assert "daily_loss_limit" in config_dict
        assert "capital_cap" in config_dict
        assert "max_position_size" in config_dict
        
        # Check that values are correct
        assert config_dict["kelly_fraction"] == config.kelly_fraction
        assert config_dict["conf_threshold"] == config.conf_threshold
        
    def test_from_optimization_method(self):
        """Test creating configuration from optimization parameters."""
        optimization_params = {
            "kelly_fraction": 0.5,
            "conf_threshold": 0.8,
            "daily_loss_limit": 0.04,
        }
        
        config = TradingConfig.from_optimization(optimization_params)
        
        # Check that optimization parameters were applied
        assert config.kelly_fraction == 0.5
        assert config.conf_threshold == 0.8
        assert config.daily_loss_limit == 0.04
        
    def test_environment_variable_override(self):
        """Test that environment variables can override default values."""
        # Set environment variable
        os.environ["KELLY_FRACTION"] = "0.35"
        os.environ["CONF_THRESHOLD"] = "0.72"
        
        try:
            config = TradingConfig.from_env("balanced")
            
            # Check that environment variables were used
            assert config.kelly_fraction == 0.35
            assert config.conf_threshold == 0.72
            
        finally:
            # Clean up environment variables
            if "KELLY_FRACTION" in os.environ:
                del os.environ["KELLY_FRACTION"]
            if "CONF_THRESHOLD" in os.environ:
                del os.environ["CONF_THRESHOLD"]
                
    def test_parameter_ranges(self):
        """Test that parameters are within reasonable ranges."""
        for mode in ["conservative", "balanced", "aggressive"]:
            config = TradingConfig.from_env(mode)
            
            # Kelly fraction should be between 0 and 1
            assert 0 <= config.kelly_fraction <= 1
            
            # Confidence threshold should be between 0 and 1
            assert 0 <= config.conf_threshold <= 1
            
            # Daily loss limit should be between 0 and 1
            assert 0 <= config.daily_loss_limit <= 1
            
            # Capital cap should be between 0 and 1
            assert 0 <= config.capital_cap <= 1
            
            # Confirmation count should be positive
            assert config.confirmation_count > 0
            
            # Position size should be positive
            assert config.max_position_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])