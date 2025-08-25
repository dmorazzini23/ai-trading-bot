from tests.optdeps import require
require("pandas")
import os

import pandas as pd

# Set test environment before importing heavy modules
os.environ['PYTEST_RUNNING'] = 'true'

import ai_trading.features.indicators as ind


def test_ensure_columns_accepts_symbol_arg():
    """Test that ensure_columns accepts optional symbol parameter"""
    df = pd.DataFrame({"close":[1,2,3]})
    out = ind.ensure_columns(df, ["macd","atr","vwap","macds"], symbol="AAPL")
    for c in ["macd","atr","vwap","macds"]:
        assert c in out.columns

def test_ensure_columns_backwards_compatible():
    """Test that ensure_columns still works with 2 arguments"""
    df = pd.DataFrame({"close":[1,2,3]})
    out = ind.ensure_columns(df, ["rsi"])
    assert "rsi" in out.columns

def test_pretrade_lookback_days_setting():
    """Test that pretrade_lookback_days setting is available"""
    from ai_trading.config.settings import get_settings
    settings = get_settings()
    assert hasattr(settings, 'pretrade_lookback_days')
    assert settings.pretrade_lookback_days == 120
