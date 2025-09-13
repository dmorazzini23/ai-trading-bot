"""Ensure package exports are explicit and stable."""


def test_root_exports():
    import ai_trading

    assert 'config' in ai_trading.__all__
    assert 'ExecutionEngine' in ai_trading.__all__


def test_config_exports():
    import ai_trading.config as config

    assert 'TradingConfig' in config.__all__
    assert 'get_env' in config.__all__


def test_core_exports():
    import ai_trading.core as core

    assert 'OrderSide' in core.__all__
    assert 'TRADING_CONSTANTS' in core.__all__


def test_utils_exports():
    import ai_trading.utils as utils

    assert 'sleep' in utils.__all__
    assert 'http' in utils.__all__

