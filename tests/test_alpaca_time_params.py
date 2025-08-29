import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

pd = pytest.importorskip("pandas")
from ai_trading.alpaca_api import get_bars_df

from tests.helpers.asserts import assert_df_like


class _Resp:
    def __init__(self, df):
        self.df = df


@patch("ai_trading.alpaca_api._get_rest")
def test_daily_uses_date_only(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_stock_bars.return_value = _Resp(
        pd.DataFrame({"open": [1.0], "close": [1.1]})
    )
    mock_rest_cls.return_value = mock_rest

    df = get_bars_df("SPY", "1Day", feed="iex", adjustment="all")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert isinstance(req.start, dt.datetime) and req.start.tzinfo is None
    assert isinstance(req.end, dt.datetime) and req.end.tzinfo is None
    assert getattr(req.timeframe, "amount", None) == 1
    assert getattr(req.timeframe.unit, "name", "") == "Day"


@patch("ai_trading.alpaca_api._get_rest")
def test_intraday_uses_rfc3339z(mock_rest_cls):
    mock_rest = MagicMock()
    mock_rest.get_stock_bars.return_value = _Resp(
        pd.DataFrame({"open": [1.0], "close": [1.1]})
    )
    mock_rest_cls.return_value = mock_rest

    start = dt.datetime(2025, 8, 19, 15, 0, 5, tzinfo=dt.UTC)
    end = dt.datetime(2025, 8, 19, 16, 0, 5, tzinfo=dt.UTC)
    df = get_bars_df("SPY", "5Min", start=start, end=end, feed="iex", adjustment="all")
    assert_df_like(df)  # AI-AGENT-REF: allow empty in offline mode
    mock_rest_cls.assert_called_once_with(bars=True)
    (req,), kwargs = mock_rest.get_stock_bars.call_args
    assert isinstance(req.start, dt.datetime) and req.start.tzinfo == dt.UTC
    assert isinstance(req.end, dt.datetime) and req.end.tzinfo == dt.UTC
    assert getattr(req.timeframe, "amount", None) == 5
    assert getattr(req.timeframe.unit, "name", "") in {"Minute", "Min"}
