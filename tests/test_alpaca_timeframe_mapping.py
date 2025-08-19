import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai_trading.alpaca_api import get_bars_df


class _Resp:
    def __init__(self, df):
        self.df = df


@patch("ai_trading.alpaca_api.TradeApiREST")
def test_day_timeframe_normalized(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_bars.return_value = _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "Day", feed="iex", adjustment="all")
    args, kwargs = mock_rest.get_bars.call_args
    assert kwargs["timeframe"] in ("1Day", "1D")
    assert not df.empty


@patch("ai_trading.alpaca_api.TradeApiREST")
def test_tf_object_normalized(mock_rest_cls):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    mock_rest = MagicMock()
    mock_rest.get_bars.return_value = _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", TimeFrame(1, TimeFrameUnit.Day), feed="iex", adjustment="all")
    args, kwargs = mock_rest.get_bars.call_args
    assert kwargs["timeframe"] in ("1Day", "1D")
    assert not df.empty


@patch("ai_trading.alpaca_api.TradeApiREST")
def test_minute_normalized(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_bars.return_value = _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "Minute", feed="iex", adjustment="all")
    args, kwargs = mock_rest.get_bars.call_args
    assert kwargs["timeframe"] in ("1Min", "1Minute")
    assert not df.empty

