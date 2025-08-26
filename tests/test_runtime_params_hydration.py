"""
Test runtime parameter hydration and validation.

Validates that TradingConfig and build_runtime ensure all required 
parameters are properly hydrated and accessible.
"""
import os
from unittest.mock import patch

import pytest


def test_trading_config_has_required_parameters():
    """Test that TradingConfig includes required trading parameters."""
    from ai_trading.config.management import TradingConfig

    cfg = TradingConfig()

    # Verify required parameters are present as attributes
    assert hasattr(cfg, 'capital_cap')
    assert hasattr(cfg, 'dollar_risk_limit')
    assert hasattr(cfg, 'max_position_size')

    # Verify default values
    assert cfg.capital_cap == 0.04
    assert cfg.dollar_risk_limit == 0.05
    assert cfg.max_position_size == 1.0


def test_trading_config_from_env_loads_parameters():
    """Test that TradingConfig.from_env() loads parameters from environment."""
    from ai_trading.config.management import TradingConfig

    # Test with environment variables
    env_vars = {
        'CAPITAL_CAP': '0.06',
        'DOLLAR_RISK_LIMIT': '0.08',
        'MAX_POSITION_SIZE': '2.0',
    }

    with patch.dict(os.environ, env_vars):
        cfg = TradingConfig.from_env()

        assert cfg.capital_cap == 0.06
        assert cfg.dollar_risk_limit == 0.08
        assert cfg.max_position_size == 2.0


def test_trading_config_from_env_market_calendar(monkeypatch):
    """TradingConfig.from_env picks up MARKET_CALENDAR."""  # AI-AGENT-REF
    from ai_trading.config.management import TradingConfig

    monkeypatch.setenv("MARKET_CALENDAR", "XNAS")
    cfg = TradingConfig.from_env()
    assert cfg.market_calendar == "XNAS"


def test_build_runtime_hydrates_all_parameters():
    """Test that build_runtime creates runtime with all required parameters."""
    from ai_trading.config.management import TradingConfig
    from ai_trading.core.runtime import REQUIRED_PARAM_DEFAULTS, build_runtime

    cfg = TradingConfig()
    runtime = build_runtime(cfg)

    # Verify runtime has params dict
    assert hasattr(runtime, 'params')
    assert isinstance(runtime.params, dict)

    # Verify all required parameters are present
    for key in REQUIRED_PARAM_DEFAULTS.keys():
        assert key in runtime.params, f"Missing required parameter: {key}"

    # Verify specific values
    assert runtime.params['CAPITAL_CAP'] == 0.04
    assert runtime.params['DOLLAR_RISK_LIMIT'] == 0.05
    assert runtime.params['MAX_POSITION_SIZE'] == 1.0


def test_build_runtime_uses_config_values():
    """Test that build_runtime uses values from TradingConfig."""
    from ai_trading.config.management import TradingConfig
    from ai_trading.core.runtime import build_runtime

    # Create config with custom values
    cfg = TradingConfig(
        capital_cap=0.08,
        dollar_risk_limit=0.10,
        max_position_size=2.5,
        kelly_fraction=0.7,
        buy_threshold=0.8,
        conf_threshold=0.9
    )

    runtime = build_runtime(cfg)

    # Verify custom values are used
    assert runtime.params['CAPITAL_CAP'] == 0.08
    assert runtime.params['DOLLAR_RISK_LIMIT'] == 0.10
    assert runtime.params['MAX_POSITION_SIZE'] == 2.5
    assert runtime.params['KELLY_FRACTION'] == 0.7
    assert runtime.params['BUY_THRESHOLD'] == 0.8
    assert runtime.params['CONF_THRESHOLD'] == 0.9


def test_param_helper_fallback_logic():
    """Test that _param helper function provides proper fallback logic."""
    from ai_trading.config.management import TradingConfig
    from ai_trading.core.runtime import build_runtime

    # Create a runtime with params
    cfg = TradingConfig(capital_cap=0.08)
    runtime = build_runtime(cfg)

    # Simulate the _param helper function logic
    def _param(runtime, key, default):
        if runtime and hasattr(runtime, 'params') and runtime.params and key in runtime.params:
            return runtime.params[key]
        if runtime and hasattr(runtime, 'cfg') and runtime.cfg:
            return float(getattr(runtime.cfg, key.lower(), default))
        return default

    # Test parameter access
    assert _param(runtime, 'CAPITAL_CAP', 0.04) == 0.08
    assert _param(runtime, 'MISSING_KEY', 999.0) == 999.0

    # Test with runtime that has no params
    class MockRuntime:
        def __init__(self, cfg):
            self.cfg = cfg

    mock_runtime = MockRuntime(cfg)
    assert _param(mock_runtime, 'CAPITAL_CAP', 0.04) == 0.08  # Falls back to cfg
    assert _param(mock_runtime, 'MISSING_KEY', 999.0) == 999.0  # Falls back to default


def test_no_missing_parameters_validation():
    """Test that validation finds no missing parameters when properly configured."""
    from ai_trading.config.management import TradingConfig
    from ai_trading.core.runtime import REQUIRED_PARAM_DEFAULTS, build_runtime

    cfg = TradingConfig.from_env()
    runtime = build_runtime(cfg)

    # Simulate the validation logic from runner.py
    missing = [k for k in REQUIRED_PARAM_DEFAULTS if k not in runtime.params]

    # Should be no missing parameters
    assert missing == [], f"Found missing parameters: {missing}"


def test_parameter_values_are_floats():
    """Test that all parameter values are properly converted to floats."""
    from ai_trading.config.management import TradingConfig
    from ai_trading.core.runtime import build_runtime

    cfg = TradingConfig()
    runtime = build_runtime(cfg)

    # All parameter values should be floats
    for key, value in runtime.params.items():
        assert isinstance(value, float), f"Parameter {key} is not a float: {type(value)}"


def test_build_runtime_ignores_none_values():
    """build_runtime should fall back to defaults when config values are None."""
    from ai_trading.config.management import TradingConfig
    from ai_trading.core.runtime import REQUIRED_PARAM_DEFAULTS, build_runtime

    cfg = TradingConfig(
        capital_cap=None,
        dollar_risk_limit=None,
        max_position_size=None,
        kelly_fraction_max=None,  # unrelated field to ensure None doesn't break
    )

    runtime = build_runtime(cfg)

    for key, default in REQUIRED_PARAM_DEFAULTS.items():
        assert runtime.params[key] == default


if __name__ == "__main__":
    pytest.main([__file__])
