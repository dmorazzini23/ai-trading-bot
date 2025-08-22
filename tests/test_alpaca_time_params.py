import datetime as dt
from unittest.mock import MagicMock, patch

import pandas as pd
from ai_trading.alpaca_api import get_bars_df

from tests.helpers.asserts import assert_df_like


class _Resp:
    def __init__(self, df):
        self.df = df


@patch("ai_trading.alpaca_api.TradeApiREST")
def test_daily_uses_date_only(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_bars.return_value = _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "Day", feed="iex", adjustment="all")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
    _, kwargs = mock_rest.get_bars.call_args
    assert (
        isinstance(kwargs["start"], str)
        and len(kwargs["start"]) == 10
        and kwargs["start"].count("-") == 2
    )
    assert (
        isinstance(kwargs["end"], str)
        and len(kwargs["end"]) == 10
        and kwargs["end"].count("-") == 2
    )
    assert kwargs["timeframe"] in ("1Day", "1D")


@patch("ai_trading.alpaca_api.TradeApiREST")
def test_intraday_uses_rfc3339z(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_bars.return_value = _Resp(pd.DataFrame({"open": [1.0], "close": [1.1]}))
    mock_rest_cls.return_value = mock_rest

    start = dt.datetime(2025, 8, 19, 15, 0, 5, tzinfo=dt.UTC)
    end = dt.datetime(2025, 8, 19, 16, 0, 5, tzinfo=dt.UTC)
    df = get_bars_df("SPY", "5Min", start=start, end=end, feed="iex", adjustment="all")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
    _, kwargs = mock_rest.get_bars.call_args
    assert kwargs["start"].endswith("Z") and "T" in kwargs["start"] and "." not in kwargs["start"]
    assert kwargs["end"].endswith("Z") and "T" in kwargs["end"] and "." not in kwargs["end"]
    assert kwargs["timeframe"] in ("5Min",)
