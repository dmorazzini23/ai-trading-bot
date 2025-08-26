import importlib

import pytest
import ai_trading.data_fetcher as df
import ai_trading.config.management as config

CURRENT_MODULES = [
    "ai_trading.data_fetcher",
    "ai_trading.config.management",
]


@pytest.mark.unit
def test_current_modules_importable():
    for name in CURRENT_MODULES:
        assert importlib.import_module(name) is not None


@pytest.mark.unit
def test_data_fetcher_active_exports():
    expected = ("get_bars", "get_bars_batch", "get_minute_df")
    for attr in expected:
        assert hasattr(df, attr)


@pytest.mark.unit
def test_config_management_exports():
    for attr in ("get_env", "reload_env", "SEED"):
        assert hasattr(config, attr)
