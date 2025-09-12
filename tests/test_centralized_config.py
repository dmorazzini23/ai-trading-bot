#!/usr/bin/env python3
"""
Test suite for centralized trading configuration system.

This test validates that the centralized TradingConfig system works correctly
across all trading modes and parameter categories.
"""

import os

import pytest

os.environ["TESTING"] = "1"
os.environ["MAX_DRAWDOWN_THRESHOLD"] = "0.2"
os.environ.pop("MAX_POSITION_SIZE", None)

from ai_trading.config import TradingConfig
from ai_trading.core.bot_engine import BotMode


class TestCentralizedConfig:
    """Test the centralized trading configuration system."""

    def test_trading_config_initialization(self, monkeypatch):
        """Test that TradingConfig initializes correctly from environment."""
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        config = TradingConfig.from_env()

        # Test default values are set
        assert config.kelly_fraction == 0.6
        assert config.daily_loss_limit == 0.03
        assert config.capital_cap == 0.04
        assert config.max_position_size == 8000

    def test_trading_config_from_env(self):
        """Test loading configuration from environment with type casting."""
        os.environ["BUY_THRESHOLD"] = "0.5"
        os.environ["SIGNAL_PERIOD"] = "10"
        try:
            config = TradingConfig.from_env()
        finally:
            del os.environ["BUY_THRESHOLD"]
            del os.environ["SIGNAL_PERIOD"]

        # Should load successfully with proper defaults and casting
        assert isinstance(config, TradingConfig)
        assert config.kelly_fraction == 0.6
        assert config.max_drawdown_threshold == 0.2
        assert config.buy_threshold == 0.5
        assert config.signal_period == 10

    def test_missing_drawdown_threshold_raises(self, monkeypatch):
        """Critical fields must be present."""
        monkeypatch.delenv("MAX_DRAWDOWN_THRESHOLD", raising=False)
        with pytest.raises(RuntimeError):
            TradingConfig.from_env({})
        monkeypatch.setenv("MAX_DRAWDOWN_THRESHOLD", "0.2")

    def test_mode_specific_configurations(self):
        """Test that each mode has appropriate parameter values."""
        conservative = TradingConfig.from_env("conservative")
        balanced = TradingConfig.from_env("balanced")
        aggressive = TradingConfig.from_env("aggressive")

        # Conservative mode should have lower risk parameters
        assert conservative.kelly_fraction < balanced.kelly_fraction
        assert conservative.conf_threshold > balanced.conf_threshold
        assert conservative.confirmation_count > balanced.confirmation_count

        # Aggressive mode should have higher risk parameters
        assert aggressive.kelly_fraction > balanced.kelly_fraction
        assert aggressive.conf_threshold < balanced.conf_threshold
        assert aggressive.confirmation_count < balanced.confirmation_count

    def test_conservative_mode_parameters(self, monkeypatch):
        """Test conservative mode specific values."""
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        config = TradingConfig.from_env("conservative")

        assert config.kelly_fraction == 0.25
        assert config.conf_threshold == 0.85
        assert config.daily_loss_limit == 0.03
        assert config.capital_cap == 0.04
        assert config.confirmation_count == 3
        assert config.take_profit_factor == 1.5
        assert config.max_position_size == 5000

    def test_balanced_mode_parameters(self, monkeypatch):
        """Test balanced mode specific values."""
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        config = TradingConfig.from_env("balanced")

        assert config.kelly_fraction == 0.6
        assert config.conf_threshold == 0.75
        assert config.daily_loss_limit == 0.03
        assert config.capital_cap == 0.04
        assert config.confirmation_count == 2
        assert config.take_profit_factor == 1.8
        assert config.max_position_size == 8000

    def test_aggressive_mode_parameters(self, monkeypatch):
        """Test aggressive mode specific values."""
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        config = TradingConfig.from_env("aggressive")

        assert config.kelly_fraction == 0.75
        assert config.conf_threshold == 0.65
        assert config.daily_loss_limit == 0.03
        assert config.capital_cap == 0.04
        assert config.confirmation_count == 1
        assert config.take_profit_factor == 2.5
        assert config.max_position_size == 12000

    def test_modern_parameter_access(self):
        """TradingConfig exposes modern field names."""
        env = {
            "CAPITAL_CAP": "0.5",
            "DOLLAR_RISK_LIMIT": "0.2",
            "MAX_POSITION_SIZE": "1000",
            "MAX_DRAWDOWN_THRESHOLD": "0.2",
        }
        config = TradingConfig.from_env(env)
        assert config.capital_cap == 0.5
        assert config.dollar_risk_limit == 0.2
        assert config.max_position_size == 1000
        assert not hasattr(config, "get_legacy_params")

    def test_bot_mode_integration(self):
        """Test that BotMode class integrates correctly with centralized config."""
        BotMode("conservative")
        BotMode("balanced")
        BotMode("aggressive")

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


def test_trading_config_has_max_drawdown_threshold():
    """TradingConfig exposes drawdown threshold."""  # AI-AGENT-REF
    from ai_trading.config.management import TradingConfig

    cfg = TradingConfig.from_env("balanced")
    assert hasattr(cfg, "max_drawdown_threshold"), "TradingConfig missing max_drawdown_threshold"
    assert isinstance(cfg.max_drawdown_threshold, int | float)
    assert 0 <= cfg.max_drawdown_threshold <= 1, "max_drawdown_threshold should be a fraction (0..1)"


def test_trading_config_to_dict_includes_capital_and_drawdown():
    """to_dict includes capital cap and drawdown."""  # AI-AGENT-REF
    from ai_trading.config.management import TradingConfig

    cfg = TradingConfig.from_env("balanced")
    data = cfg.to_dict()
    assert "capital_cap" in data, "to_dict() missing capital_cap"
    assert "max_drawdown_threshold" in data, "to_dict() missing max_drawdown_threshold"
    assert isinstance(data["capital_cap"], int | float)
    assert isinstance(data["max_drawdown_threshold"], int | float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
